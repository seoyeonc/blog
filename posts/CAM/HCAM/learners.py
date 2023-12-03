import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# torch
import torchvision
import torch
from torchvision import transforms

from fastai.vision.all import *

import rpy2
import rpy2.robjects as ro 
from rpy2.robjects.vectors import FloatVector 
from rpy2.robjects.packages import importr


from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, EigenGradCAM, LayerCAM

import cv2

from .utils import Img_loader

        
def Other_method(lrnr, *args, status='cpu', cam_method='gradcam',input_img=None):
    md = lrnr.model.to(status)
    target_layer = lrnr.model[0][-1]
    if cam_method == 'gradcam':
        other_cam = GradCAM(model=md, target_layers=target_layer)
    elif cam_method == 'hirescam':
        other_cam = HiResCAM(model=md, target_layers=target_layer)
    elif cam_method == 'gradcamplusplus':
        other_cam = GradCAMPlusPlus(model=md, target_layers=target_layer)
    elif cam_method == 'ablationcam':
        other_cam = AblationCAM(model=md, target_layers=target_layer)
    elif cam_method == 'xgradcam':
        other_cam = XGradCAM(model=md, target_layers=target_layer)
    elif cam_method == 'eigencam':
        other_cam = EigenCAM(model=md, target_layers=target_layer)
    elif cam_method == 'fullgrad':
        other_cam = FullGrad(model=md, target_layers=target_layer)
    elif cam_method == 'eigengradcam':
        other_cam = EigenGradCAM(model=md, target_layers=target_layer)
    elif cam_method == 'layercam':
        other_cam = LayerCAM(model=md, target_layers=target_layer)
    else:
        raise ValueError(f"Invalid CAM method: {cam_method}")

    cam_ret = other_cam(input_tensor=input_img, targets=None)
    return cam_ret

class HCAM:
    def __init__(self,lrnr):
        self.md = lrnr.model
        self.net1 = self.md[0]
        self.net2 = self.md[1]

    def learner_thresh(self,Thresh=1600,input_img=None):
        self.net1.to('cpu')
        self.net2.to('cpu')

        camimg = torch.einsum('ij,jkl -> ikl', self.net2[2].weight, self.net1(input_img).squeeze())
        ebayesthresh = importr('EbayesThresh').ebayesthresh

        power_threshed=np.array(ebayesthresh(FloatVector(torch.tensor(camimg[0].detach().reshape(-1))**2)))
        self.ybar_threshed = np.where(power_threshed>Thresh,torch.tensor(camimg[0].detach().reshape(-1)),0)
        self.ybar_threshed = torch.tensor(self.ybar_threshed.reshape(16,16))

    def learner_step(self,Rate=-0.05):
        self.A1 = torch.exp(Rate*(self.ybar_threshed))
        self.A2 = 1 - self.A1

    def prob(self,input_img=None):
        self.a,self.b = self.md(input_img).tolist()[0]
        self.a_prob = np.exp(self.a)/ (np.exp(self.a)+np.exp(self.b))
        self.b_prob = np.exp(self.b)/ (np.exp(self.a)+np.exp(self.b))

    def mode_decomp(self,input_img=None):
        # mode res
        X_res=np.array(self.A1.to("cpu").detach(),dtype=np.float32)
        Y_res=torch.Tensor(cv2.resize(X_res,(512,512),interpolation=cv2.INTER_LINEAR))

        self.x_res=(input_img.squeeze().to('cpu')-torch.min(input_img.squeeze().to('cpu')))*Y_res
        self.x_res = self.x_res.reshape(1,3,512,512)

        # mode
        X=np.array(self.A2.to("cpu").detach(),dtype=np.float32)
        Y=torch.Tensor(cv2.resize(X,(512,512),interpolation=cv2.INTER_LINEAR))

        self.x=(input_img.squeeze().to('cpu')-torch.min(input_img.squeeze().to('cpu')))*Y
        
    def __call__(self,input_img=None):
        A1 = self.A1
        A2 = self.A2
        x = self.x
        x_res = self.x_res
        return {'A1':A1, 'A2':A2, 'x':x, 'x_res':x_res} 



