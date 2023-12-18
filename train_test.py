import od
import random
import numpy as np
import torch
import time
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import RandomSampler
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
from utils import compute_avg_metrics


def train(dataloaders, model, optimizer, scheduler, args, logger):
    cudnn.benchmark = False
    cudnn.deterministic = True
    train_loader, eval_loader, test_loader = dataloaders
    model.train()
    cls_criterion = nn.CrossEntropyLoss()
    start = time.time()
    print(model)
    print("Number of Trainable Parameters: %d" % count_parameters(model))

    use_patch, roi_dir = ('_patch_', 'all_st_patches_512') if args.use_features else ('_', 'all_st')
    cur_iter = 0
    for epoch in range(args.epochs):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        for i, (img, gene, grade) in enumerate(train_loader):
            img, gene, grade = img.cuda(non_blocking=True), gene.cuda(non_blocking=True), grade.cuda(non_blocking=True)
            features, pred = model(img)

            # classification loss
            cls_loss = cls_criterion(pred, grade)
            # TODO: add regression loss
            loss = cls_loss

            if args.rank == 0:
                train_loss = loss.item()
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if dist.is_available() and dist.is_initialized():
                loss = loss.data.clone()
                dist.all_reduce(loss.div_(dist.get_world_size()))

            cur_iter += 1
            if args.rank == 0:
                if cur_iter % 30 == 1:
                    cur_lr = optimizer.param_groups[0]['lr']
                    val_acc, val_f1, val_auc, val_bac, val_sens, val_spec, val_prec, val_mcc, val_kappa = validate(eval_loader, model)
                    test_acc, test_f1, test_auc, test_bac, test_sens, test_spec, test_prec, test_mcc, test_kappa = validate(test_loader, model)
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
                                'validation': {'Accuracy': val_acc,
                                                'F1 score': val_f1,
                                                'AUC': val_auc,
                                                'Balanced Accuracy': val_bac,
                                                'Sensitivity': val_sens,
                                                'Specificity': val_spec,
                                                'Precision': val_prec,
                                                'MCC': val_mcc,
                                                'Kappa': val_kappa},
                                'train': {'loss': train_loss,
                                            'learning_rate': cur_lr}},)
                
                    print('\rEpoch: [%2d/%2d] Iter [%4d/%4d] || Time: %4.4f sec || lr: %.6f || Loss: %.4f' % (
                        epoch, args.epochs, i + 1, len(train_loader), time.time() - start,
                        cur_lr, loss.item()), end='', flush=True)
            if args.task == "surv":
                risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))   # Logging Information
                censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))   # Logging Information
                survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))   # Logging Information
            elif args.task == "grad":
                pred = pred.argmax(dim=1, keepdim=True)
                grad_acc_epoch += pred.eq(grade.view_as(pred)).sum().item()
            
            if args.verbose > 0 and args.print_every > 0 and (batch_idx % args.print_every == 0 or batch_idx+1 == len(train_loader)):
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch+1, args.niter+args.niter_decay, batch_idx+1, len(train_loader), loss.item()))


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