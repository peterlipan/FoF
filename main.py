import os
import torch
import wandb
import pickle
import argparse
import torch.distributed as dist
from models import CreateModel
from datasets import TCGADataset
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from torch.utils.data import DataLoader
from utils import yaml_config_hook, convert_model, train


def main(gpu, args, wandb_logger):
    if gpu != 0:
        wandb_logger = None

    rank = args.nr * args.gpus + gpu
    args.rank = rank
    args.device = rank

    if args.world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load data file
    data_cv = pickle.load(open(args.data_file, 'rb'))
    # TODO: implement cross-validation
    data_cv_split = data_cv['split'][1]

    # training set
    train_dataset = TCGADataset(args, split='train')

    # set sampler for parallel training
    if args.world_size > 1:
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
    )
    if rank == 0:
        test_dataset = TCGADataset(args, split='test')
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        test_loader = None

    loaders = (train_loader, test_loader)

    num_classes = train_dataset.num_classes

    # model init
    model = CreateModel(backbone=args.backbone, num_classes=num_classes, hid_dim=args.hidden_dim, pretrained=args.pretrained)
    if args.reload:
        model_fp = os.path.join(
            args.checkpoints, "epoch_{}_.pth".format(args.epochs)
        )
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))

    model = model.to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9)
    # TODO: implement scheduler
    scheduler = None

    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)
        ema_model = convert_model(ema_model)
        ema_model = DataLoader(ema_model)
    else:
        if args.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu])

    train(loaders, model, optimizer, scheduler, args, wandb_logger)

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    yaml_config = yaml_config_hook("./config/configs.yaml")
    for k, v in yaml_config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    parser.add_argument('--debug', action="store_true", help='debug mode(disable wandb)')
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    # Master address for distributed data parallel
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # check checkpoints path
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)

    # init wandb
    if not args.debug:
        wandb.login(key="cb1e7d54d21d9080b46d2b1ae2a13d895770aa29")
        config = dict()

        for k, v in yaml_config.items():
            config[k] = v

        wandb_logger = wandb.init(
            project="MCL_{:s}".format(args.dataset),
            config=config
        )
    else:
        wandb_logger = None

    if args.world_size > 1:
        print(
            f"Training with {args.world_size} GPUS, waiting until all processes join before starting training"
        )
        mp.spawn(main, args=(args, wandb_logger,), nprocs=args.world_size, join=True)
    else:
        main(0, args, wandb_logger)