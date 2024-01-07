import torch
import time
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
from .metrics import compute_avg_metrics
from .losses import GeneGuidance, RegionContrastiveLoss
from .cam_helper import get_swin_cam, cam2mask


def train(dataloaders, model, optimizer, scheduler, args, logger):
    cudnn.benchmark = False
    cudnn.deterministic = True
    train_loader, test_loader = dataloaders
    model.train()
    cls_criterion = nn.CrossEntropyLoss()
    start = time.time()
    gene_guidance = GeneGuidance(args.batch_size, args.world_size)
    global_local = RegionContrastiveLoss(args.batch_size, args.temperature, args.world_size, args.dataparallel)
    cur_iter = 0
    hidden_size = model.module.config.hidden_size if isinstance(model, DataParallel) or isinstance(model, DDP) else model.config.hidden_size
    patch_size = model.module.config.patch_size if isinstance(model, DataParallel) or isinstance(model, DDP) else model.config.patch_size
    projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, 64, bias=False),
        )
    projector = projector.cuda()
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        for i, (img, gene, grade) in enumerate(train_loader):
            img, gene, grade = img.cuda(non_blocking=True), gene.cuda(non_blocking=True), grade.cuda(non_blocking=True)
            
            if epoch > args.warmup_epochs:
                # Class activation map
                cam = get_swin_cam(model, img, grade, smooth=True)
                mask = cam2mask(cam, patch_size=patch_size, threshold=args.threshold)

                # global-local consistency
                model.zero_grad()
                features, pred = model(img)
                pos_features, _ = model(img, token_mask=mask)
                neg_features, _ = model(img, token_mask=~mask)
                anchor_contrast, pos_contrast, neg_contrast = projector(features), projector(pos_features), projector(neg_features)
                region_loss = args.lambda_region * global_local(anchor_contrast, pos_contrast, neg_contrast)
                # classification loss
                cls_loss = cls_criterion(pred, grade)
                gene_loss = args.lambda_gene * (gene_guidance(features, gene) + gene_guidance(pos_features, gene)) 
                loss = cls_loss + gene_loss + region_loss
            else:
                features, pred = model(img)
                cls_loss = cls_criterion(pred, grade)
                loss = cls_loss

            if args.rank == 0:
                train_loss = loss.item()
                cls_loss_item = cls_loss.item()
                gene_loss_item = gene_loss.item() if epoch > args.warmup_epochs else 0
                region_loss_item = region_loss.item() if epoch > args.warmup_epochs else 0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if dist.is_available() and dist.is_initialized():
                loss = loss.data.clone()
                dist.all_reduce(loss.div_(dist.get_world_size()))

            cur_iter += 1
            if args.rank == 0:
                if cur_iter % 10 == 0:
                    cur_lr = optimizer.param_groups[0]['lr']
                    test_acc, test_f1, test_auc, test_bac, test_sens, test_spec, test_prec, test_mcc, test_kappa = validate(
                        test_loader, model)
                    if logger is not None:
                        logger.log({'test': {'Accuracy': test_acc,
                                             'F1 score': test_f1,
                                             'AUC': test_auc,
                                             'Balanced Accuracy': test_bac,
                                             'Sensitivity': test_sens,
                                             'Specificity': test_spec,
                                             'Precision': test_prec,
                                             'MCC': test_mcc,
                                             'Kappa': test_kappa},
                                    'train': {'loss': train_loss,
                                              'cls_loss': cls_loss_item,
                                              'gene_loss': gene_loss_item,
                                              'region_loss': region_loss_item,
                                              'learning_rate': cur_lr}}, )

                    print('\rEpoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f' % (
                        epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                        cur_lr, loss.item()), end='', flush=True)

        scheduler.step()


def validate(dataloader, model):
    training = model.training
    model.eval()

    ground_truth = torch.Tensor().cuda()
    predictions = torch.Tensor().cuda()

    with torch.no_grad():
        for img, gene, grade in dataloader:
            img, gene, grade = img.cuda(non_blocking=True), gene.cuda(non_blocking=True), grade.cuda(non_blocking=True)
            _, pred = model(img)
            pred = F.softmax(pred, dim=1)
            ground_truth = torch.cat((ground_truth, grade))
            predictions = torch.cat((predictions, pred))

        acc, f1, auc, bac, sens, spec, prec, mcc, kappa = compute_avg_metrics(ground_truth, predictions)
        model.train(training)
        return acc, f1, auc, bac, sens, spec, prec, mcc, kappa
