import random
import torchvision
from torchvision import transforms
import torchvision.utils as vutils
import numpy as np
import torch
import time
from torch import nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
import scipy.io as scio
import kornia
from DMROFusion import GCM, Encoder_HYP, Decoder
from args import train_data_path,train_path,batch_size,lr,is_cuda,log_interval,img_size,epochs

Train_Image_Number=len(os.listdir(train_data_path+'FLIR\\'))
Iter_per_epoch=(Train_Image_Number % batch_size!=0)+Train_Image_Number//batch_size
# =============================================================================
# Preprocessing and dataset establishment  
# =============================================================================
transforms = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        ])
                          
Data = torchvision.datasets.ImageFolder(train_data_path,transform=transforms)
dataloader = torch.utils.data.DataLoader(Data, batch_size,shuffle=True)
# =============================================================================
# Models
# =============================================================================
GCM_Train=GCM()
Encoder_HPY_Train=Encoder_HYP()
Decoder_Train=Decoder()

if is_cuda:
    GCM_Train=GCM_Train.cuda()
    Encoder_HPY_Train=Encoder_HPY_Train.cuda()
    Decoder_Train=Decoder_Train.cuda()
 
 
print(GCM_Train)
print(Encoder_HPY_Train)
print(Decoder_Train)


optimizer1 = optim.Adam(GCM_Train.parameters(), lr = lr)
optimizer2 = optim.Adam(Encoder_HPY_Train.parameters(), lr =0.1* lr)
optimizer3 = optim.Adam(Decoder_Train.parameters(), lr = lr)


scheduler1 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer1, [41, 81], gamma=0.1)
scheduler2 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer2, [41, 81], gamma=0.1)
scheduler3 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer3, [41, 81], gamma=0.1)

MSELoss = nn.MSELoss()
SmoothL1Loss=nn.SmoothL1Loss()
L1Loss=nn.L1Loss()
SSIMLoss = kornia.losses.SSIMLoss(3, reduction='mean')

# =============================================================================
# Training
# =============================================================================
print('============ Training Begins ===============')
print('The total number of images is %d,\n Need to cycle %d times.'%(Train_Image_Number,Iter_per_epoch))

for iteration in range(epochs):

    GCM_Train.train()
    Encoder_HPY_Train.train()
    Decoder_Train.train()
    
   
    data_iter_input = iter(dataloader)
    img_input, _ = next(data_iter_input)
    if iteration <= 10:
        hyper_para = 10
    else:
        hyper_para = random.randint(1,20)

    for step in range(Iter_per_epoch):

          
        if is_cuda:
            img_input=img_input.cuda()
        
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        # =====================================================================
        # Calculate loss  
        # =====================================================================
        im1=GCM_Train(img_input)
        fea=Encoder_HPY_Train(img_input, im1, hyper_para*torch.ones(1,1,1,1).cuda())
        img_recon=Decoder_Train(fea)
        # Total loss
        mse=MSELoss(img_input, img_recon)
        ssim=SSIMLoss(img_input, img_recon)
        loss = mse + hyper_para*ssim
        # Update
        loss.backward()
        optimizer1.step() 
        optimizer2.step()
        optimizer3.step()
        # =====================================================================
        # Print 
        # =====================================================================
        los = loss.item()
        mse_l=mse.item()
        ssim_l=ssim.item()
        if (step + 1) % log_interval == 0:          
            print('Epoch/step: %d/%d, loss: %.7f, lr: %f' %(iteration+1, step+1, los, optimizer1.state_dict()['param_groups'][0]['lr']))
            print('MSELoss:%.7f\nSSIMLoss:%.7f'%(mse_l,ssim_l))

    scheduler1.step()
    scheduler2.step()
    scheduler3.step()

# Save models
GCM_Train.eval()
GCM_Train.cpu()
Encoder_HPY_Train.eval()
Encoder_HPY_Train.cpu()
Decoder_Train.eval()
Decoder_Train.cpu()

name=['GCM_Train','Encoder_HYP_Train','Decoder_Train']
os.mkdir('./Train_result/')
for i in range(3):
    save_model_filename =name[i] + ".model"#str(time.ctime()).replace(' ', '_').replace(':', '_') + "_" +
    save_model_path = os.path.join(train_path, save_model_filename)
    if i == 0:
        torch.save(GCM_Train.state_dict(), save_model_path)
    elif i == 1:
        torch.save(Encoder_HPY_Train.state_dict(), save_model_path)
    elif i == 2:
        torch.save(Decoder_Train.state_dict(), save_model_path)
    print("\nDone, trained model saved at", save_model_path)

 