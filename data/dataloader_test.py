import os
import torch
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from data.implement import get_trainval_dataloader, get_test_dataloader


def get_train_img_info():
    train_df = pd.read_csv(config.csv_train)
    print(train_df['label'].value_counts())
    train_df['filename'] = train_df['filename'].apply(
        lambda x: os.path.join(config.dir_raw_train, x))

    nums_img = len(train_df)

    train_mean = []
    train_std = []
    train_scale = []
    for index in tqdm(range(nums_img)):
        img = Image.open(train_df['filename'].iloc[index]).convert('RGB')
        img = np.array(img)

        # plt.imshow(img)
        # plt.show()

        batch_mean = np.mean(img, axis=(0, 1))
        batch_std = np.std(img, axis=(0, 1))
        train_mean.append(batch_mean)
        train_std.append(batch_std)
        train_scale.append(img.shape[0] / img.shape[1])

    train_mean = torch.tensor(np.mean(train_mean, axis=0))
    train_std = torch.tensor(np.mean(train_std, axis=0))
    # train_scale = torch.tensor(np.mean(train_scale, axis=0))
    print('Mean:', train_mean)
    print('Std Dev:', train_std)
    # print('Img Scale:', train_scale)

    plt.hist(train_scale, bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.show()


def dataloader_test(dataloader):
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(dataloader):
            # print("Epoch: ",epoch," Step:",step," batch_x",batch_x.numpy(),"batch_y:",batch_y.numpy())
            print("单个batch的size: ", batch_x.shape)
            print("单个batch的label张量: ", batch_y.numpy())

            img = batch_x[0].numpy()
            img = np.transpose(img, (1, 2, 0))

            plt.imshow(img)
            plt.show()
            break
        break


if __name__ == "__main__":
    # get_train_img_info()
    # test_dataloader = get_test_dataloader()
    # dataloader_test(test_dataloader)
    train_dataloader, valid_dataloader = get_trainval_dataloader()
    dataloader_test(train_dataloader)
