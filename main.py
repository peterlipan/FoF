import os
import torch
import wandb
import pickle
import argparse
import torch.distributed as dist
from models import Transformer, ContrastiveProjectors
from dataset import TCGADataset, generate_splits
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup
from utils import yaml_config_hook, convert_model, train
import warnings


def main(gpu, args, wandb_logger):
    if gpu != 0:
        wandb_logger = None

    rank = args.nr * args.gpus + gpu
    args.rank = rank
    args.device = rank

    if args.world_size > 1 and not args.dataparallel:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load data file
    data_cv = pickle.load(open(args.data_path, 'rb'))
    data_cv_split = data_cv['splits'][args.fold]
    gene_names = data_cv['data_pd'].columns[-80:]

    # training set
    train_dataset = TCGADataset(args, data_cv_split, gene_names, split='train')
    step_per_epoch = len(train_dataset) // (args.batch_size * args.world_size)

    # set sampler for parallel training
    if args.world_size > 1 and not args.dataparallel:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
    )
    if rank == 0:
        test_dataset = TCGADataset(args, data_cv_split, gene_names, split='test')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    else:
        test_loader = None

    loaders = (train_loader, test_loader)

    num_classes = train_dataset.num_classes

    # model init
    model = Transformer(image_size=args.image_size, num_classes=num_classes, 
                            pretrained=args.pretrained, patch_size=args.patch_size)
    global_projectors = ContrastiveProjectors(model.config.hidden_size, args.dis_gene, teacher=True)
    local_projectors = ContrastiveProjectors(model.config.hidden_size, args.dis_gene, teacher=False)


    model = model.cuda()
    global_projectors = global_projectors.cuda()
    local_projectors = local_projectors.cuda()

    optim_params = [{'params': model.parameters()}, {'params': local_projectors.parameters(), 'lr_mult': 10}]
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_epochs * step_per_epoch, args.epochs * step_per_epoch)

    if args.dataparallel:
        model = convert_model(model)
        local_projectors = convert_model(local_projectors)
        model = DataParallel(model, device_ids=[int(x) for x in args.visible_gpus.split(",")])
        local_projectors = DataParallel(local_projectors, device_ids=[int(x) for x in args.visible_gpus.split(",")])

    else:
        if args.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            local_projectors = torch.nn.SyncBatchNorm.convert_sync_batchnorm(local_projectors)
            model = DDP(model, device_ids=[gpu], find_unused_parameters=True, broadcast_buffers=False)
            local_projectors = DDP(local_projectors, device_ids=[gpu], find_unused_parameters=True, broadcast_buffers=False)
    
    models = (model, global_projectors, local_projectors)
    train(loaders, models, optimizer, scheduler, args, wandb_logger)


if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    yaml_config = yaml_config_hook("./config/configs.yaml")
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    parser.add_argument('--debug', action="store_true", help='debug mode(disable wandb)')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes

    # split the dataset based on gbmlgg15cv_all_st_0_0_0.pkl provided by pathomics
    # no need to redo if you have my_split_dropGradeNaN.pkl in the dataset folder
    # generate_splits(args.data_path, args.seed, subset=args.split)

    # Master address for distributed data parallel
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # check checkpoints path
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    # init wandb
    if not args.debug:
        wandb.login(key="YOUR WANDB KEY HERE")
        config = dict()

        for k, v in yaml_config.items():
            config[k] = v

        wandb_logger = wandb.init(
            project="MCL_{:s}".format(args.dataset),
            config=config
        )
    else:
        wandb_logger = None

    if args.world_size > 1 and not args.dataparallel:
        print(
            f"Training with {args.world_size} GPUS, waiting until all processes join before starting training"
        )
        mp.spawn(main, args=(args, wandb_logger,), nprocs=args.world_size, join=True)
    else:
        main(0, args, wandb_logger)
