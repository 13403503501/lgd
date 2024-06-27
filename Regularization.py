import torch.nn as nn
import torch.utils.tensorboard
import torchvision 
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
# 正则化
class Regularization(nn.Module):
    def __init__(self,model,weight_decay,p=2):
        super(Regularization, self).__init__()
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    # 前向传播
    def forward(self,model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list,self.weight_decay,p=self.p)
        return reg_loss
    
    # 确定设备
    def to(self, device):
        self.device = device
        super().to(device)
        return self
    
    # 得到正则化参数
    def get_weight(self, model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
    
    # 求取正则化的和
    def regularization_loss(self, weight_list, weight_decay, p=2):
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=self.p)
            reg_loss = reg_loss + l2_reg
        reg_loss = weight_decay * reg_loss
        return reg_loss
    
    # 打印有哪些参数
    def weight_info(self, weight_list):
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")
