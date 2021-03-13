import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import ttach as tta
import torch
from torch.utils.data import DataLoader
import config
from model import Net

from data import MyDataSet, transform_test, int2label

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict(model):
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch_x, _ in tqdm(test_loader):
            batch_x = batch_x.to(device)
            probs = model(batch_x)
            pred_list.extend(probs.cpu().numpy())
    return pred_list


def single_model_predict():
    assert len(model_name_list) == 1
    model_name = model_name_list[0]
    model = Net(model_name).to(device)
    model_save_path = os.path.join(
        config.dir_weight, '{}.bin'.format(model_name))
    model.load_state_dict(torch.load(model_save_path))

    pred_list = []
    with torch.no_grad():
        for batch_x, _ in tqdm(test_loader):
            batch_x = batch_x.to(device)
            probs = model(batch_x)
            probs = torch.max(torch.softmax(probs, dim=1), dim=1)
            probs = probs[1].cpu().numpy()
            pred_list += probs.tolist()

    submission = pd.DataFrame({
        "id": range(len(pred_list)),
        "label": [int2label(x) for x in pred_list]
    })
    submission.to_csv(config.dir_csv_test, index=False, header=False)


def single_model_predict_tta():
    assert len(model_name_list) == 1
    model_name = model_name_list[0]
    model = Net(model_name).to(device)
    model_save_path = os.path.join(
        config.dir_weight, '{}.bin'.format(model_name))
    model.load_state_dict(torch.load(model_save_path))

    transforms = tta.Compose([
        tta.HorizontalFlip(),
        # tta.Rotate90(angles=[0, 180]),
        # tta.Scale(scales=[1, 2, 4]),
        # tta.Multiply(factors=[0.9, 1, 1.1]),
        tta.FiveCrops(224, 224)
    ])

    tta_model = tta.ClassificationTTAWrapper(model, transforms)

    pred_list = []
    with torch.no_grad():
        for batch_x, _ in tqdm(test_loader):
            batch_x = batch_x.to(device)
            probs = tta_model(batch_x)
            probs = torch.max(torch.softmax(probs, dim=1), dim=1)
            probs = probs[1].cpu().numpy()
            pred_list += probs.tolist()

    submission = pd.DataFrame({
        "id": range(len(pred_list)),
        "label": [int2label(x) for x in pred_list]
    })
    submission.to_csv(config.dir_csv_test, index=False, header=False)


def multi_model_predict():
    preds_dict = dict()
    for model_name in model_name_list:
        for fold_idx in range(5):
            model = Net(model_name).to(device)
            model_save_path = os.path.join(
                config.dir_weight, '{}_fold{}.bin'.format(model_name, fold_idx))
            model.load_state_dict(torch.load(model_save_path))
            pred_list = predict(model)
            submission = pd.DataFrame(pred_list)
            # submission = pd.DataFrame({"id": range(len(pred_list)), "label": pred_list})
            submission.to_csv(
                '{}/{}_fold{}_submission.csv'.format(config.dir_submission, config.save_model_name, fold_idx),
                index=False,
                header=False)
            preds_dict['{}_{}'.format(model_name, fold_idx)] = pred_list
    pred_list = get_pred_list(preds_dict)
    submission = pd.DataFrame(
        {"id": range(len(pred_list)), "label": [int2label(x) for x in pred_list]})
    submission.to_csv(config.dir_csv_test, index=False, header=False)


def multi_model_predict_tta():
    preds_dict = dict()
    for model_name in model_name_list:
        for fold_idx in range(5):
            model = Net(model_name).to(device)
            model_save_path = os.path.join(
                config.dir_weight, '{}_fold{}.bin'.format(model_name, fold_idx))
            model.load_state_dict(torch.load(model_save_path))
            '/home/muyun99/data/dataset/AIyanxishe/Image_Classification/weight/resnet18_train_size_256_fold0.bin'
            transforms = tta.Compose([
                tta.Resize([int(config.size_test_image), int(config.size_test_image)]),
                tta.HorizontalFlip(),
                # tta.Rotate90(angles=[0, 180]),
                # tta.Scale(scales=[1, 2, 4]),
                # tta.Multiply(factors=[0.9, 1, 1.1]),
                tta.FiveCrops(config.size_test_image, config.size_test_image)
            ])
            tta_model = tta.ClassificationTTAWrapper(model, transforms)

            pred_list = predict(tta_model)
            submission = pd.DataFrame(pred_list)
            submission.to_csv(
                '{}/{}_fold{}_submission.csv'.format(config.dir_submission, config.save_model_name, fold_idx),
                index=False,
                header=False
            )
            preds_dict['{}_{}'.format(model_name, fold_idx)] = pred_list

    pred_list = get_pred_list(preds_dict)
    submission = pd.DataFrame(
        {"id": range(len(pred_list)), "label": [int2label(x) for x in pred_list]})
    submission.to_csv(config.dir_csv_test, index=False, header=False)


def file2submission():
    preds_dict = dict()
    for model_name in model_name_list:
        for fold_idx in range(5):
            df = pd.read_csv('{}/{}_fold{}_submission.csv'
                             .format(config.dir_submission, model_name, fold_idx), header=None)
            preds_dict['{}_{}'.format(model_name, fold_idx)] = df.values
    pred_list = get_pred_list(preds_dict)
    submission = pd.DataFrame(
        {"id": range(len(pred_list)), "label": [int2label(x) for x in pred_list]})
    submission.to_csv(config.dir_csv_test, index=False, header=False)


def get_pred_list(preds_dict):
    pred_list = []
    if predict_mode == 1:
        for i in range(data_len):
            preds = []
            for model_name in model_name_list:
                for fold_idx in range(5):
                    prob = preds_dict['{}_{}'.format(model_name, fold_idx)][i]
                    pred = np.argmax(prob)
                    preds.append(pred)
            pred_list.append(max(preds, key=preds.count))
    else:
        for i in range(data_len):
            prob = None
            for model_name in model_name_list:
                for fold_idx in range(5):
                    if prob is None:
                        prob = preds_dict['{}_{}'.format(
                            model_name, fold_idx)][i] * ratio_dict[model_name]
                    else:
                        prob += preds_dict['{}_{}'.format(
                            model_name, fold_idx)][i] * ratio_dict[model_name]
            pred_list.append(np.argmax(prob))
    return pred_list


if __name__ == "__main__":
    model_name_list = config.predict_model_names.split('+')
    ratio_list = config.predict_ratios = '1'.split(',')
    predict_mode = config.predict_mode

    ratio_dict = dict()
    for i, ratio in enumerate(ratio_list):
        ratio_dict[model_name_list[i]] = int(ratio)

    data_len = len(os.listdir(config.dir_raw_test))
    test_path_list = [
        '{}/{}.jpg'.format(config.dir_raw_test, x) for x in range(0, data_len)]
    test_data = np.array(test_path_list)
    test_dataset = MyDataSet(test_data, transform_test, 'test')
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False)

    # single_model_predict()
    # single_model_predict_tta()
    # multi_model_predict_tta()
    file2submission()
