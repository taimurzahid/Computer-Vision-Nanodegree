## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


# https://render.githubusercontent.com/view/ipynb?commit=edc7c622114d20105eda9ec8b2298018d8abde36&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f7461696d75727a616869642f446565702d4c6561726e696e672d4e616e6f6465677265652f656463376336323231313464323031303565646139656338623232393830313864386162646533362f70726f6a6563742d646f672d636c617373696669636174696f6e2f646f675f6170702e6970796e62&nwo=taimurzahid%2FDeep-Learning-Nanodegree&path=project-dog-classification%2Fdog_app.ipynb&repository_id=166848379&repository_type=Repository#(IMPLEMENTATION)-Model-Architecture

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)
        
        # (224 - 5)/1 + 1 = 220
        # output = (32, 220, 220)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1_batchnorm = nn.BatchNorm2d(32)
        
        # 220/2 = 110
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # (110 - 3)/1 + 1 = 108
        # output = (64, 108, 108)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv2_batchnorm = nn.BatchNorm2d(64)
        
        # 108/2 = 54
        # output = (64, 54, 54)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # (54 - 3)/1 + 1 = 52
        # output = (128, 52, 52)
        #self.conv3 = nn.Conv2d(64, 128, 3)
        
        # 52/2 = 26
        # output = (128, 26, 26)
        #self.pool3 = nn.MaxPool2d(2, 2)
        
        # (26 - 3)/1 + 1 = 24
        # output = (256, 24, 24)
        #self.conv4 = nn.Conv2d(128, 256, 3)
        
        # 24/2 = 12
        #self.pool4 = nn.MaxPool2d(2, 2)
        
        # (12 - 1)/1 + 1 = 12
        #self.conv5 = nn.Conv2d(256, 512, 1)
        # 12/2 = 6
        #self.pool5 = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(64 * 54 * 54, 1024)
        self.fc1_batchnorm = nn.BatchNorm2d(1024)
        
        self.fc2 = nn.Linear(1024, 256)
        self.fc2_batchnorm = nn.BatchNorm2d(256)
        
        #self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 136)
        
        self.dropout1 = nn.Dropout(0.20)
        self.dropout2 = nn.Dropout(0.15)
        self.dropout3 = nn.Dropout(0.20)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        #img = img.view(1, -1)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        #x = self.pool3(F.relu(self.conv3(x)))
        #x = self.pool4(F.relu(self.conv4(x)))
        #x = self.pool5(F.relu(self.conv5(x)))
        x = self.dropout1(x) 
        
        x = x.view(x.size(0), -1) # flatten image input
        #x = x.view(1, -1)
        #x = x.view(-1, 512 * 6 * 6)
        
        x = self.fc1(x) # add 1st hidden layer, with
        x = F.relu(x) # relu activation function
        
        x = self.dropout1(x) # add dropout layer

        x = self.fc2(x) # add 2nd hidden layer
        x = F.relu(x) # relu activation function
        
        #x = self.dropout2(x) # add dropout layer
        #x = self.fc3(x) # add 3rd hidden layer
        #x = F.relu(x) # relu activation function
        
        x = self.dropout3(x) # add dropout layer
        x = self.fc4(x) # add 4th hidden layer
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
