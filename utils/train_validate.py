import cv2
import torch
import wandb
import time
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
from .metrics import compute_avg_metrics
from .losses import GeneGuidance, RegionContrastiveLoss, MultiHeadContrastiveLoss
from .cam_helper import get_swin_cam, cam2mask
from pytorch_grad_cam.utils.image import show_cam_on_image


def update_ema_variables(global_model, local_model, alpha, step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (step + 1), alpha)
    for ema_param, param in zip(global_model.module.swin.parameters(), local_model.module.swin.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(dataloaders, models, optimizer, scheduler, args, logger):
    cudnn.benchmark = False
    cudnn.deterministic = True
    train_loader, test_loader = dataloaders
    global_model, local_model = models
    global_model.train()
    local_model.train()
    cls_criterion = nn.CrossEntropyLoss()
    start = time.time()
    
    cur_iter = 0
    hidden_size = global_model.module.config.hidden_size if isinstance(global_model, DataParallel) or isinstance(global_model, DDP) else global_model.config.hidden_size
    float_gene_guidance = GeneGuidance(args.batch_size, args.world_size)
    discrete_gene_guidance = MultiHeadContrastiveLoss(args.batch_size, args.world_size, hidden_size, args.dis_gene)
    global_local = RegionContrastiveLoss(args.batch_size, args.temperature, args.world_size, hidden_size)
    neg_grade = torch.zeros(args.batch_size, requires_grad=False).long().cuda()
    # label the negative regions as grade 0
    for epoch in range(args.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        for i, (img, dis_gene, float_gene, grade) in enumerate(train_loader):
            img, dis_gene, float_gene, grade = img.cuda(non_blocking=True), dis_gene.cuda(non_blocking=True), \
                                               float_gene.cuda(non_blocking=True), grade.cuda(non_blocking=True)

            # Class activation map
            cam = get_swin_cam(global_model, img, grade, smooth=True)
            mask = cam2mask(cam, patch_size=args.patch_size, threshold=args.threshold)

            # global-local consistency
            global_model.zero_grad()
            features, pred = global_model(img)
            pos_features, pos_pred = local_model(img, token_mask=mask)
            neg_features, neg_pred = local_model(img, token_mask=~mask)
            region_loss = args.lambda_region * global_local(features, pos_features, neg_features)
            # classification loss
            # global grade: [0, 2]; local grade: [0, 3] where 0 is the dummy/normal class
            cls_loss = (cls_criterion(pos_pred, grade + 1) + cls_criterion(pred, grade) + cls_criterion(neg_pred, neg_grade)) / 3
        
            global_pos_features = torch.cat((pos_features, features), dim=0)
            global_pos_gene = torch.cat((float_gene, float_gene), dim=0)
            dis_gene_loss = args.lambda_dis_gene * discrete_gene_guidance(features, pos_features, neg_features, dis_gene)
            float_gene_loss = args.lambda_float_gene * float_gene_guidance(global_pos_features, global_pos_gene)
            
            loss = cls_loss
            if epoch >= args.warmup:
                loss += dis_gene_loss + float_gene_loss + region_loss

            if args.rank == 0:
                train_loss = loss.item()
                cls_loss_item = cls_loss.item()
                float_gene_loss_item = float_gene_loss.item()
                dis_gene_loss_item = dis_gene_loss.item()
                region_loss_item = region_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update the ema model
            update_ema_variables(global_model, local_model, args.ema_decay, cur_iter)

            if dist.is_available() and dist.is_initialized():
                loss = loss.data.clone()
                dist.all_reduce(loss.div_(dist.get_world_size()))

            cur_iter += 1
            if args.rank == 0:
                if cur_iter % 50 == 0 and logger is not None:
                    # pick 3 images from a batch
                    wandb_imgs = img.permute(0,2,3,1).detach().cpu().numpy()[:4]
                    wandb_imgs = [(item - np.min(item)) / np.ptp(item) for item in wandb_imgs]
                    wandb_cams = cam.detach().cpu().numpy()[:5]
                    # resize the images and cams to 224x224
                    wandb_imgs = [cv2.resize(item, (224, 224)) for item in wandb_imgs]
                    wandb_cams = [cv2.resize(item, (224, 224)) for item in wandb_cams]
                    img_cam = [show_cam_on_image(img, cam, use_rgb=True) for img, cam in zip(wandb_imgs, wandb_cams)]
                    logger.log({'Image with CAM': [wandb.Image(item) for item in img_cam],
                    'Original Image': [wandb.Image(item) for item in wandb_imgs]})
                if cur_iter % 10 == 0:
                    cur_lr = optimizer.param_groups[0]['lr']
                    test_acc, test_f1, test_auc, test_bac, test_sens, test_spec, test_prec, test_mcc, test_kappa = validate(
                        test_loader, global_model)
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
                                              'float_gene_loss': float_gene_loss_item,
                                                'dis_gene_loss': dis_gene_loss_item,
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
        for img, _, _, grade in dataloader:
            img, grade = img.cuda(non_blocking=True), grade.cuda(non_blocking=True)
            _, pred = model(img)
            pred = F.softmax(pred, dim=1)
            ground_truth = torch.cat((ground_truth, grade))
            predictions = torch.cat((predictions, pred))

        acc, f1, auc, bac, sens, spec, prec, mcc, kappa = compute_avg_metrics(ground_truth, predictions)
    model.train(training)
    return acc, f1, auc, bac, sens, spec, prec, mcc, kappa
