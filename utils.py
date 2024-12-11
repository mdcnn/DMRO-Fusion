import numpy as np
import torch
from DMROFusion import GCM, Encoder_HYP, Decoder
import torch.nn.functional as F

device='cuda'

def output_img(x):
    return x.cpu().detach().numpy()[0,0,:,:]

def l1_addition(y1,y2,window_width=1):
      ActivityMap1 = y1.abs()
      ActivityMap2 = y2.abs()

      kernel = torch.ones(2*window_width+1,2*window_width+1)/(2*window_width+1)**2
      kernel = kernel.to(device).type(torch.float32)[None,None,:,:]
      kernel = kernel.expand(y1.shape[1],y1.shape[1],2*window_width+1,2*window_width+1)
      ActivityMap1 = F.conv2d(ActivityMap1, kernel, padding=window_width)
      ActivityMap2 = F.conv2d(ActivityMap2, kernel, padding=window_width)
      WeightMap1 = ActivityMap1/(ActivityMap1+ActivityMap2)
      WeightMap2 = ActivityMap2/(ActivityMap1+ActivityMap2)
      return WeightMap1*y1+WeightMap2*y2

def Test_fusion(img_test1, img_test2, addition_mode='Sum', hyper_para=5):
    GCM_Test = GCM().to(device)
    GCM_Test.load_state_dict(torch.load(
            "./Train_result/GCM_Train.model"
            ))
    Encoder_HYP_Test = Encoder_HYP().to(device)
    Encoder_HYP_Test.load_state_dict(torch.load(
            "./Train_result/Encoder_HYP_Train.model"
            ))
    Decoder_Test = Decoder().to(device)
    Decoder_Test.load_state_dict(torch.load(
            "./Train_result/Decoder_Train.model"
            ))
    
    GCM_Test.eval()
    Encoder_HYP_Test.eval()
    Decoder_Test.eval()
    
    img_test1 = np.array(img_test1, dtype='float32')/255
    img_test1 = img_test1[0:int(np.floor(img_test1.shape[0]/8) * 8), 0:int(np.floor(img_test1.shape[1]/8) * 8)]
    print(img_test1.shape)
    img_test1 = torch.from_numpy(img_test1.reshape((1, 1, img_test1.shape[0], img_test1.shape[1])))
    
    img_test2 = np.array(img_test2, dtype='float32')/255
    img_test2 = img_test2[0:int(np.floor(img_test2.shape[0]/8) * 8), 0:int(np.floor(img_test2.shape[1]/8) * 8)]
    print(img_test2.shape)
    img_test2 = torch.from_numpy(img_test2.reshape((1, 1, img_test2.shape[0], img_test2.shape[1])))


    
    img_test1=img_test1.cuda()
    img_test2=img_test2.cuda()

    with torch.no_grad():
        B_K_IR=GCM_Test(img_test1)
        B_K_VIS=GCM_Test(img_test2)
        D_K_IR_=Encoder_HYP_Test(img_test1, B_K_IR, hyper_para*torch.ones(1,1,1,1).cuda())
        D_K_VIS_=Encoder_HYP_Test(img_test2, B_K_VIS, hyper_para*torch.ones(1,1,1,1).cuda())
        
    if addition_mode=='Sum':      
        F_b=(D_K_IR_+D_K_VIS_)

    elif addition_mode=='Average':
        F_b=(D_K_IR_+D_K_VIS_)/2

    elif addition_mode=='l1_norm':
        F_b=l1_addition(D_K_IR_, D_K_VIS_)
        
    with torch.no_grad():
        Out = Decoder_Test(F_b)
     
    return output_img(Out)#(Out)
