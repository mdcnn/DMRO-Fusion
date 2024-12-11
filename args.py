Train_data_choose='FLIR'
if Train_data_choose=='FLIR':
    train_data_path = '.\\Datasets\\Train_data_FLIR\\'
    log_interval = 12
    epochs = 80

train_path = '.\\Train_result\\'
lr = 1*1e-2
is_cuda = True
img_size=128
batch_size=32




