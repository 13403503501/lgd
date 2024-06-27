import torch
from torch import nn
 
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
 
        # [b, 784] => [b, 20]
        self.encoder1 = nn.Sequential(
            nn.Linear(784, 256),
            nn.Sigmoid()
        )

        self.encoder2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()
        )
 
        # [b, 20] => [b, 784]
        self.decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )
 
 
    def forward(self, x, return_feature=False):
        """
        :param x: [b, 1, 28, 28]
        :return:
        """
        batchsz = x.size(0)
        # flatten（打平）
        x = x.view(batchsz, 784)
        # encoder1
        feature = self.encoder1(x)
        x= self.encoder2(feature)
        # decoder
        x = self.decoder(x)
        # reshape
        x = x.view(batchsz, 1, 28, 28)
        if return_feature== True:
            return x,feature
        return x


    