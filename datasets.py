import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
import torchvision.transforms as T
from torch.utils.data.dataset import Dataset
from albumentations.pytorch import ToTensorV2


class TCGADataset(Dataset):
    def __init__(self, args, data, gene_list, split='train'):
        # selected_genes = ['codeletion', 'idh mutation', 'EGFR', 'CDKN2A', 'CDKN2B', 'PTEN', 'MDM4', 'MYC', 'RB1', 'FGFR2', 'BRAF', '7p', '7q', '9p', '10q']
        # selected by ANOVA test
        self.dis_gene = args.dis_gene
        self.float_gene = args.float_gene
        # find the corresponding index of the selected genes
        dis_gene_idx = [gene_list.tolist().index(gene) for gene in self.dis_gene]
        float_gene_idx = [gene_list.tolist().index(gene) for gene in self.float_gene]
        self.img = data[split]['x_path']
        self.dis_gene = data[split]['x_omic'][:, dis_gene_idx]
        self.float_gene = data[split]['x_omic'][:, float_gene_idx]
        self.split = split
        self.grade = data[split]['grade']
        self.num_classes = len(set(self.grade))

        self.spatial_transform = A.Compose(
            [
                A.Resize(height=args.image_size, width=args.image_size),
                A.HorizontalFlip(p=.5),
                A.VerticalFlip(p=.5),
                A.RandomRotate90(p=.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.5),
                A.OneOf([
                    A.RandomGridShuffle(grid=(3, 3), p=.5),
                    A.RandomGridShuffle(grid=(7, 7), p=.5),
                    A.RandomGridShuffle(grid=(11, 11), p=.5),
                ], p=.5),
            ]
        )

        self.color_transform = A.Compose(
            [
                A.ColorJitter(p=.5),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.ChannelShuffle(p=.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        
        self.test_transform = A.Compose([
            A.Resize(height=args.image_size, width=args.image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
            ])

    def __len__(self):
        return len(self.grade)
        
    def __getitem__(self, index):
        img = np.array(Image.open(self.img[index]).convert('RGB'))
        dis_gene = torch.tensor(self.dis_gene[index]).long()
        float_gene = torch.tensor(self.float_gene[index]).float()
        grade = torch.tensor(self.grade[index]).long()
        if self.split == 'train':
            common = self.spatial_transform(image=img)['image']
            view1 = self.color_transform(image=common)['image']
            view2 = self.color_transform(image=common)['image']
            view3 = self.color_transform(image=common)['image']
            return view1, view2, view3, dis_gene, float_gene, grade
        else:
            img = self.test_transform(image=img)['image']
            return img, dis_gene, float_gene, grade
        