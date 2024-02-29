import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from typing import List


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)


def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


class CIFAR10_net(nn.Module):
    def __init__(self):
        super(CIFAR10_net, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)

        # Fully connected layer
        # Note: The number of input features to this layer depends on the output size of the conv2 layer
        self.fc1 = nn.Linear(64 * 5 * 5, 512)  # Assuming the input image size is 32x32

        # Output layer
        self.fc2 = nn.Linear(512, 10)  # 10 classes for CIFAR-10

    def forward(self, x):
        # Apply first convolutional layer and pooling
        x = self.pool(F.relu(self.conv1(x)))

        # Apply second convolutional layer and pooling
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output for the fully connected layer
        x = x.view(-1, 64 * 5 * 5)  # Flatten

        # Apply fully connected layer with ReLU
        x = F.relu(self.fc1(x))

        # Apply output layer
        x = self.fc2(x)
        return x





class CIFAR100_net(nn.Module):
    def __init__(self):
        super(CIFAR100_net, self).__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.dropout = nn.Dropout(0.3)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.dropout3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 100)  # 100 classes for CIFAR-100

    def forward(self, x):
        # Convolutional layers with ReLU, BatchNorm, MaxPool and Dropout
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        # x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))
        # x = self.dropout(x)

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1)

        # Fully connected layer with ReLU and Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)

        # Output layer
        x = self.fc2(x)
        return x


class STL10_net(nn.Module):
    def __init__(self):
        super(STL10_net, self).__init__()
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 12 * 12, 1024)  # Adjusted for the reduced image size
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)  # 10 classes for STL-10

    def forward(self, x):
        # Apply first conv layer with pooling
        x = self.pool1(F.relu(self.conv1(x)))

        # Apply second conv layer with pooling
        x = self.pool2(F.relu(self.conv2(x)))

        # Apply third conv layer with pooling
        x = self.pool3(F.relu(self.conv3(x)))

        # Flatten the output for the fully connected layers
        x = x.view(-1, 256 * 12 * 12)  # Flatten

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer
        x = self.fc3(x)
        return x




class LeNet5_CIFAR10(nn.Module):
    def __init__(self):
        super(LeNet5_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.pool = nn.MaxPool2d(2, 2)  
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)    

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5_FashionMNIST(nn.Module):
    def __init__(self):
        super(LeNet5_FashionMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x








class ResNet9_new(nn.Module):
    def __init__(self, in_channels, num_classes,dim=1028):
        super(ResNet9_new, self).__init__()
        
        self.conv1 = self._make_layer(in_channels, 64)
        self.conv2 = self._make_layer(64, 128, pool=True) 
        self.res1 = nn.Sequential(self._make_layer(128, 128), self._make_layer(128, 128)) 
        
        self.conv3 = self._make_layer(128, 256, pool=True)
        self.conv4 = self._make_layer(256, 512, pool=True) 
        self.res2 = nn.Sequential(self._make_layer(512, 512), self._make_layer(512, 512)) 
        self.conv5 = self._make_layer(512, 1028, pool=True) 
        self.res3 = nn.Sequential(self._make_layer(1028, 1028), self._make_layer(1028, 1028))  
        
        self.classifier = nn.Sequential(nn.MaxPool2d(2), # 1028 x 1 x 1
                                        nn.Flatten(), # 1028 
                                        nn.Linear(dim, num_classes)) # 1028 -> 100

    def _make_layer(self, in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.conv5(out)
        out = self.res3(out) + out
        out = self.classifier(out)
        return out





class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes, dim=4608):
        super(ResNet9, self).__init__()
        self.prep = self._make_layer(in_channels, 64)
        self.layer1_head = self._make_layer(64, 128, pool=True)
        self.layer1_residual = nn.Sequential(
            self._make_layer(128, 128), 
            self._make_layer(128, 128)
        )
        self.layer2 = self._make_layer(128, 256, pool=True)
        self.layer3_head = self._make_layer(256, 512, pool=True)
        self.layer3_residual = nn.Sequential(
            self._make_layer(512, 512), 
            self._make_layer(512, 512)
        )
        self.MaxPool2d = nn.MaxPool2d(4)
        self.linear = nn.Linear(dim, num_classes)

    def _make_layer(self, in_channels, out_channels, pool=False):
        layers = [
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1_head(x)
        x = self.layer1_residual(x) + x
        
        x = self.layer2(x)
        x = self.layer3_head(x)
        x = self.layer3_residual(x) + x
        
        x = self.MaxPool2d(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.linear(x)
        return x


def weight_init(m):
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    return 





"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])




def get_model(dataset,arch):
    if dataset == "CIFAR10":
        if arch == "CNN":
            model = CIFAR10_net()
        elif arch == "LeNet":
            model = LeNet5_CIFAR10()
        elif arch == "Resnet9":
            model = ResNet9(in_channels=3,num_classes=10,dim=512)
    if dataset == "F-MNIST":
        if arch == "CNN":
            model = FashionMnist_net()
        elif arch == "LeNet":
            model = LeNet5_FashionMNIST()

    elif dataset == "CIFAR100":
        if arch == "CNN":
            model = CIFAR100_net()
        elif arch == "Resnet9":
            model = ResNet9(in_channels=3,num_classes=100,dim=512)
        elif arch == "Resnet9_new":
            model = ResNet9_new(in_channels=3,num_classes=100,dim=1028)
        elif arch == "Resnet18":
            model = resnet18()
        elif arch == "Resnet34":
            model = resnet34()
    elif dataset == "STL10":
        if arch == "CNN":
            model = STL10_net()
        elif arch == "Resnet9":
            model = ResNet9(in_channels=3,num_classes=10,dim=4608)
        elif arch == "Resnet9_new":
            model = ResNet9_new(in_channels=3,num_classes=10,dim=9252)
        # elif arch = "Resnet18":
        #     model = resnet18()

    # for layer in model.children():
    #     if hasattr(layer, 'reset_parameters'):
    #         layer.reset_parameters()
    
    # model.apply(weight_init)
    return model
