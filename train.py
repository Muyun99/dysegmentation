import gc
import os
import time
import torch
import config
import pandas as pd
import torch.nn as nn
from model import UNet, DeepLabV3_plus
from ranger import Ranger
from utils import accuracy
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from data import MyDataSet, transform_train, transform_valid, get_trainval_dataloader, get_test_dataloader
from metric import eval_net_unet_miou, eval_net_unet_bfscore, eval_net_unet_dice
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(fold_idx=None):
    # model = UNet(n_classes=1, n_channels=3)
    model = DeepLabV3_plus(num_classes=1, backbone='resnet', sync_bn=True)
    train_dataloader, valid_dataloader = get_trainval_dataloader()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Ranger(model.parameters(), lr=1e-3, weight_decay=0.0005)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_val_score = 0
    last_improved_epoch = 0
    if fold_idx is None:
        print('start')
        model_save_path = os.path.join(config.dir_weight, '{}.bin'.format(config.save_model_name))
    else:
        print('start fold: {}'.format(fold_idx + 1))
        model_save_path = os.path.join(config.dir_weight, '{}_fold{}.bin'.format(config.save_model_name, fold_idx))
    for cur_epoch in range(config.num_epochs):
        start_time = int(time.time())
        model.train()
        print('epoch: ', cur_epoch + 1)
        cur_step = 0
        for batch in train_dataloader:
            batch_x = batch['image']
            batch_y = batch['mask']
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            mask_pred = model(batch_x)
            train_loss = criterion(mask_pred, batch_y)
            train_loss.backward()
            optimizer.step()

            cur_step += 1
            if cur_step % config.step_train_print == 0:
                train_acc = accuracy(mask_pred, batch_y)
                msg = 'the current step: {0}/{1}, train loss: {2:>5.2}, train acc: {3:>6.2%}'
                print(msg.format(cur_step, len(train_dataloader), train_loss.item(), train_acc[0].item()))

        val_miou = eval_net_unet_miou(model, valid_dataloader, device)
        val_score = val_miou
        if val_score > best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), model_save_path)
            improved_str = '*'
            last_improved_epoch = cur_epoch
        else:
            improved_str = ''
        msg = 'the current epoch: {0}/{1}, val score: {3:>6.2%}, cost: {4}s {5}'
        end_time = int(time.time())
        print(msg.format(cur_epoch + 1, config.num_epochs, val_score,
                         end_time - start_time, improved_str))
        if cur_epoch - last_improved_epoch > config.num_patience_epoch:
            print("No optimization for a long time, auto-stopping...")
            break
        scheduler_cosine.step()
    del model
    gc.collect()
    return best_val_score

if __name__ == "__main__":
    torch.manual_seed(config.seed_random)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    sum_val_acc = 0
    sum_val_loss = 0

    best_val_score = train()
    print("best_val_score:" + str(best_val_score))

