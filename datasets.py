import os
import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data.dataset import Dataset


class TCGADataset(Dataset):
    def __init__(self, args, data, gene_list, split='train'):
        # selected_genes = ['codeletion', 'idh mutation', 'EGFR', 'CDKN2A', 'CDKN2B', 'PTEN', 'MDM4', 'MYC', 'RB1', 'FGFR2', 'BRAF', '7p', '7q', '9p', '10q']
        # selected by ANOVA test
        selected_genes = args.gene
        # find the corresponding index of the selected genes
        gene_idx = [gene_list.tolist().index(gene) for gene in selected_genes]
        self.img = data[split]['x_path']
        self.gene = data[split]['x_omic'][:, gene_idx]
        self.split = split
        self.grade = data[split]['grade']
        self.num_classes = len(set(self.grade))
        
        self.train_transform = T.Compose([
            T.Resize(args.image_size),
            T.RandomHorizontalFlip(.5),
            T.RandomVerticalFlip(.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        self.test_transform = T.Compose([
            T.Resize(args.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.grade)
        
    def __getitem__(self, index):
        img = Image.open(self.img[index]).convert('RGB')
        if self.split == 'train':
            img = self.train_transform(img)
        else:
            img = self.test_transform(img)
        gene = torch.tensor(self.gene[index]).float()
        grade = torch.tensor(self.grade[index]).long()
        return img, gene, grade
        
