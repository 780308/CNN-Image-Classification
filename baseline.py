import torch
import torch.nn as nn
import torch.nn.functional as F

    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__() 

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)




    def forward(self,x):
        # todo
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
        
from torchsummary import summary

class CIFAR10Model(nn.Module):
    def __init__(self):
        super(CIFAR10Model, self).__init__()
        # Block 1
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn0 = nn.BatchNorm2d(32)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(0.2)
        
        # Block 2
        self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(0.3)
        
        # Block 3
        self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout(0.4)

        # Block 4
        self.conv4_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout(0.5)
        
        # Flatten layer
        self.flatten = nn.Flatten()
        
        # Dense layer
        self.fc1 = nn.Linear(256 * 2 * 2, 512)  
        self.drop_fc1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512,256)
        self.drop_fc2 = nn.Dropout(0.5)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1_1(x))
        x = self.bn0(x)
        x = F.relu(self.conv1_2(x))
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        # Block 2
        x = F.relu(self.conv2_1(x))
        x = self.bn2(x)
        x = F.relu(self.conv2_2(x))
        x = self.bn3(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        # Block 3
        x = F.relu(self.conv3_1(x))
        x = self.bn4(x)
        x = F.relu(self.conv3_2(x))
        x = self.bn5(x)
        x = self.pool3(x)
        x = self.drop3(x)

        # Block 4
        x = F.relu(self.conv4_1(x))
        x = self.bn6(x)
        x = F.relu(self.conv4_2(x))
        x = self.bn7(x)
        x = self.pool4(x)
        x = self.drop4(x)
        
        # Flatten
        x = self.flatten(x)
        
        x = F.relu(self.fc1(x))
        x = self.drop_fc1(x)
        x = F.relu(self.fc2(x))
        x = self.drop_fc2(x)
        x = self.out(x)
        
        return x

# 检查模型
model = CIFAR10Model()
summary(model, (3, 32, 32))