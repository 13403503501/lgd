import torch
from torch import nn

class CNNAE(nn.Module):
    def __init__(self):
        super(CNNAE, self).__init__()
        # Encoder
        self.Encoder = nn.Sequential(
            # param [input_c, output_c, kernel_size, stride, padding]
            nn.Conv2d(1, 64, 3, 1, 1),   # [, 64, 96, 96]
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1), # [, 64, 96, 96]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),             # [, 64, 48, 48]
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1),  # [, 64, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, 1, 1), # [, 128, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, 1, 1), # [, 128, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, 1, 1), # [, 256, 48, 48]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                 # [, 256, 24, 24]
            nn.BatchNorm2d(256)   
        )
        
        # decoder
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3 ,1, 1),   # [, 128, 24, 24]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, 3, 2, 1, 1),   # [, 128, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),    # [, 64, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 3, 1, 1),      # [, 32, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, 1, 1),      # [, 32, 48, 48]
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),  # [, 16, 96, 96]
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, 1, 1),         # [, 3, 96, 96]
            nn.Sigmoid()
        )
    
    def forward(self, x, return_feature=False):
        encoder = self.Encoder(x)
        decoder = self.Decoder(encoder)
        return decoder
        
# # 输出网络结构
# DAEmodel = DenoiseAutoEncoder()