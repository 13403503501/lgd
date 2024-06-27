import torchvision
from torch.utils.data import DataLoader
import numpy
import random,os
import PIL.Image as Image
import torchvision.transforms as transforms
from torch import nn,optim
import torch
from model import AE
from tqdm import tqdm
from my_parser import set_parser
from dataset import build_dataset
import torch.utils.tensorboard

import matplotlib.pyplot as plt
args = set_parser()
args.dataset='KMNIST'
train_datasets, test_datasets = build_dataset(args)
 
train_loader = DataLoader(train_datasets, batch_size=100, shuffle=True)
test_loader = DataLoader(test_datasets, batch_size=100, shuffle=False)
 
#模型，优化器，损失函数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=AE().to(device)
criteon = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
 
##导入预训练模型
# if os.path.exists('./MNIST_Gaussian.pth') :
# if os.path.exists('./MNIST_Pepper.pth') :
# if os.path.exists('./KMNIST_Gaussian.pth') :
if os.path.exists('./KMNIST_Gaussian2.pth') :
    # 如果存在已保存的权重，则加载
    checkpoint = torch.load('KMNIST_Gaussian2.pth',map_location=lambda storage,loc:storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    initepoch = checkpoint['epoch']
    loss = checkpoint['loss']
else:
    initepoch=0

writer = torch.utils.tensorboard.SummaryWriter(args.dataset+"dir")
# args.noise_type = 'Pepper'
#开始测试
model.eval()

with torch.no_grad(): # 不求导
    step = 0
    
    for true_input,target in test_loader:
        
        #生成均值为0，方差为0.3的高斯分布
        if args.noise_type=='Gaussian':
            noise_tensor=torch.normal(mean=0,std=0.3,size=true_input.shape)
        else:# without noise
            noise_tensor = torch.zeros(size=true_input.shape)
            
        image_noise=true_input+noise_tensor
        if args.noise_type=='Pepper':
            noise_rand = torch.rand(size=true_input.shape)
            #添加椒盐噪声
            image_noise[noise_rand <0.1]=0 #椒噪声
            image_noise[noise_rand  > (1-0.1)] = 1 #盐噪声
        true_input=true_input.to(device)
        image_noise=image_noise.to(device)
        
        #限制像素的范围在0-1之间
        image_noise=torch.clamp(image_noise,min=0,max=1)
        # input = true_input
        input = torch.clamp(true_input,min=0,max=1)
        target = target.to(device)
        output = model(image_noise)
        # 损失值
        loss = criteon(output, true_input)
        step+=1
        print('Step: {} \MSELoss: {:.6f}'.format(
                step, loss.item()))
        # tensorboard
        writer.add_images("minst_test_input",input[0:5],step)
        writer.add_images("minst_test_noise",image_noise[0:5],step)
        writer.add_images("minst_test_output",output[0:5],step)
        if(step%50 ==0):
            fig = plt.figure()
            len = true_input.shape[3]
            plt.subplot(1, 3, 1)
            plt.imshow(image_noise[0].view(len,len,-1).cpu().numpy(), cmap='gray')
            plt.subplot(1, 3, 2)
            plt.imshow(output[0].view(len,len,-1).cpu().numpy(), cmap='gray')
            plt.subplot(1, 3, 3)
            plt.imshow(true_input[0].view(len,len,-1).cpu().numpy(), cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.show()
    
    writer.close()

