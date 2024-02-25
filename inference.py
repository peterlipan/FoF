import os
import cv2
import torch
import pickle
import numpy as np
from PIL import Image
import albumentations as A
from models import Transformer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from albumentations.pytorch import ToTensorV2
from utils.cam_helper import get_swin_cam
from pytorch_grad_cam.utils.image import show_cam_on_image


class TCGADataset4Inf(Dataset):
    def __init__(self, data, image_size=1024):
        self.img = np.concatenate([data['train']['x_path'], data['test']['x_path']])
        self.grade = np.concatenate([data['train']['grade'], data['test']['grade']])
        self.num_classes = len(set(self.grade))

        self.test_transform = A.Compose([
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.grade)

    def __getitem__(self, index):
        img = np.array(Image.open(self.img[index]).convert('RGB'))
        grade = torch.tensor(self.grade[index]).long()
        img = self.test_transform(image=img)['image']
        return img, grade


def save_img(img, cam, root, label, idx):
    # define the path
    label = label.detach().cpu().numpy()[0]
    path = os.path.join(root, f"grade_{label}")
    os.makedirs(path, exist_ok=True)

    # to numpy
    img = img.permute(0, 2, 3, 1).detach().cpu().numpy()[0]
    img = (img - np.min(img)) / np.ptp(img)
    cam = cam.detach().cpu().numpy()[0]
    pixel_mask = cam > 0.5
    pos_region = img * pixel_mask[..., np.newaxis]
    img_with_cam = show_cam_on_image(img, cam, use_rgb=False)

    # save the image
    cv2.imwrite(os.path.join(path, f"{idx}_cam.jpg"), img_with_cam)
    cv2.imwrite(os.path.join(path, f"{idx}_pos.jpg"), np.uint8(255 * pos_region))
    cv2.imwrite(os.path.join(path, f"{idx}_img.jpg"), np.uint8(255 * img))


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    save_path = "./results/Ours"
    os.makedirs(save_path, exist_ok=True)

    # load data file
    data_cv = pickle.load(open("./dataset/my_split_dropGradeNaN.pkl", 'rb'))
    data_cv_split = data_cv['splits'][0]

    # dataset
    dataset = TCGADataset4Inf(data_cv_split)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # model init
    model = Transformer(image_size=1024, num_classes=dataset.num_classes, pretrained="WinKawaks/vit-tiny-patch16-224", patch_size=16)
    # load model
    checkpoint = torch.load("./weights/fold_0.pth")
    model.load_state_dict(checkpoint)
    model = model.cuda()

    # inference
    idx = 0
    for img, grade in loader:
        print(f"\rProcessing {idx}th image", end='', flush=True)
        img, grade = img.cuda(non_blocking=True), grade.cuda(non_blocking=True)
        cam = get_swin_cam(model, img, grade, smooth=True)
        save_img(img, cam, save_path, grade, idx)
        idx += 1


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    main()
