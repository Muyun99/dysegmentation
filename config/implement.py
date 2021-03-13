import os
dir_raw_train = "/home/muyun99/data/dataset/AIyanxishe/Image_Classification/train/"
dir_raw_test = "/home/muyun99/data/dataset/AIyanxishe/Image_Classification/test/"
dir_submission = "/home/muyun99/data/dataset/AIyanxishe/Image_Classification/submission"
dir_csv_train = '/home/muyun99/data/dataset/AIyanxishe/Image_Classification/train.csv'
dir_csv_test = os.path.join(dir_submission, 'test.csv')
dir_weight = '/home/muyun99/data/dataset/AIyanxishe/Image_Classification/weight'


seed_random = 2020

num_classes = 6
num_epochs = 100
num_patience_epoch = 10
num_KFold = 5

size_valid = 0.1
step_train_print = 200

factor_train = 1.25
size_train_image = 256
size_valid_image = 256
size_test_image = 256

batch_size = 8


predict_mode = 1

model_name = 'se_resnext50'
save_model_name = 'se_resnext50'
predict_model_names = "se_resnext50+resnest50+efficientnet-b4"
