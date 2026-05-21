import torch
from torch import nn


class RFFI(nn.Module):
    def __init__(self):
        super(RFFI, self).__init__()

        self.Conv1 = nn.Conv2d(1, 12, kernel_size=2, padding=0)
        self.Conv2 = nn.Conv2d(12, 24, kernel_size=2, padding=1)
        self.Conv3 = nn.Conv2d(24, 48, kernel_size=2, padding=1)
        self.Conv4 = nn.Conv2d(48, 48, kernel_size=2, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(num_features=12)
        self.batchnorm2 = nn.BatchNorm2d(num_features=24)
        self.batchnorm3 = nn.BatchNorm2d(num_features=48)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, padding=1)
        # 输入宽度256/8
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1,32))
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3264, 1024),
            nn.ReLU(),
            # nn.Dropout(p=0.5),
            nn.Linear(1024, 120),
            nn.ReLU(),
            # nn.Dropout(p=0.3),
            
            # 修改分类数量
            nn.Linear(120, 40),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.Conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.Conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.Conv3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)

        '''
        x = self.Conv4(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        
        x = self.Conv4(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        '''

        x = self.maxpool(x)

        x = self.fc(x)

        return x