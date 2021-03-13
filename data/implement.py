import logging

import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

import config

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

transform_train = transforms.Compose([
    transforms.Resize([321, 321]),
    transforms.ToTensor(),
])

transform_valid = transforms.Compose([
    transforms.Resize([321, 321]),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize([321, 321]),
    transforms.ToTensor(),
])


class MyDataSet(Dataset):
    def __init__(self, file_csv, transform):
        super(MyDataSet, self).__init__()
        self.df = pd.read_csv(file_csv)
        self.transform = transform
        logging.info(f'Creating dataset with {len(self.df)} examples')

    def __getitem__(self, index):
        img = Image.open(self.df.iloc[index, 0]).convert('RGB')
        img = self.transform(img)

        mask = Image.open(self.df.iloc[index, 1]).convert('L')
        mask = self.transform(mask)

        return {
            'image': img,
            'mask': mask,
        }

    def __len__(self):
        return len(self.df)


def get_trainval_dataloader():
    train_dataset = MyDataSet(
        file_csv='/home/muyun99/MyGithub/dysegmentation/data/train_clean.csv', transform=transform_train)
    valid_dataset = MyDataSet(file_csv="/home/muyun99/MyGithub/dysegmentation/data/valid_clean.csv", transform=transform_valid)

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(
        valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    return train_dataloader, val_dataloader


def get_test_dataloader():
    test_dataset = MyDataSet(file_csv="/home/muyun99/MyGithub/dysegmentation/data/test.csv", transform=transform_test)

    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    return test_dataloader


def dataloader_test(dataloader):
    for epoch in range(3):
        for step, batch in enumerate(dataloader):
            imgs = batch['image']
            true_masks = batch['mask']
            print("单个img的size: ", imgs.shape)
            print("单个mask的size: ", true_masks.shape)

            img = imgs[0].numpy()
            img = np.transpose(img, (1, 2, 0))

            mask = true_masks[0].numpy()
            mask = np.transpose(mask, (1, 2, 0))

            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.subplot(1, 2, 2)
            plt.imshow(mask)
            plt.show()
            break
        break


if __name__ == '__main__':
    train_dataloader, valid_dataloader = get_trainval_dataloader()
    dataloader_test(train_dataloader)
