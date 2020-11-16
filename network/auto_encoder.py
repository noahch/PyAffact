import torch.nn as nn
import torch.nn.functional as F


# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, padding=3, stride=1)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1,  stride=2)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1,  stride=2)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1,  stride=2)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1,  stride=2)
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv6 = nn.Conv2d(16, 8, kernel_size=3, padding=1,  stride=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        # self.pool = nn.MaxPool2d(2, 2)

        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        # self.t_conv0 = nn.ConvTranspose2d(4, 4, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.t_conv1 = nn.ConvTranspose2d(8, 8, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.t_conv2 = nn.ConvTranspose2d(8, 32, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.t_conv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, output_padding=1, padding=1)
        # self.t_conv4 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, output_padding=1, padding=1)
        # self.t_conv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, output_padding=1, padding=1)
        self.t_conv6 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, output_padding=1, padding=1)

    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        # x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        # x = self.pool(x)  # compressed representation

        ## decode ##
        # add transpose conv layers, with relu activation function
        # x = F.relu(self.t_conv0(x))
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        # output layer (with sigmoid for scaling from 0 to 1)
        # x = F.relu(self.t_conv4(x))
        # x = F.relu(self.t_conv5(x))
        x = F.sigmoid(self.t_conv6(x))
        return x


# # define the NN architecture
# class ConvAutoencoder(nn.Module):
#     def __init__(self):
#         super(ConvAutoencoder, self).__init__()
#         ## encoder layers ##
#         # conv layer (depth from 3 --> 16), 3x3 kernels
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=7, padding=3, stride=1)
#         # conv layer (depth from 16 --> 4), 3x3 kernels
#         self.conv2 = nn.Conv2d(16, 4, kernel_size=3, padding=1,  stride=2)
#         # conv layer (depth from 16 --> 4), 3x3 kernels
#         self.conv3 = nn.Conv2d(4, 4, kernel_size=3, padding=1,  stride=2)
#         # conv layer (depth from 16 --> 4), 3x3 kernels
#         self.conv4 = nn.Conv2d(4, 4, kernel_size=3, padding=1,  stride=2)
#         # conv layer (depth from 16 --> 4), 3x3 kernels
#         self.conv5 = nn.Conv2d(4, 4, kernel_size=3, padding=1,  stride=2)
#         # pooling layer to reduce x-y dims by two; kernel and stride of 2
#         # self.pool = nn.MaxPool2d(2, 2)
#
#         ## decoder layers ##
#         ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
#         self.t_conv1 = nn.ConvTranspose2d(4, 4, kernel_size=3, stride=2, output_padding=1, padding=1)
#         self.t_conv2 = nn.ConvTranspose2d(4, 4, kernel_size=3, stride=2, output_padding=1, padding=1)
#         self.t_conv3 = nn.ConvTranspose2d(4, 4, kernel_size=3, stride=2, output_padding=1, padding=1)
#         self.t_conv4 = nn.ConvTranspose2d(4, 3, kernel_size=3, stride=2, output_padding=1, padding=1)
#
#     def forward(self, x):
#         ## encode ##
#         # add hidden layers with relu activation function
#         # and maxpooling after
#         x = F.relu(self.conv1(x))
#         # x = self.pool(x)
#         # add second hidden layer
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = F.relu(self.conv5(x))
#         # x = self.pool(x)  # compressed representation
#
#         ## decode ##
#         # add transpose conv layers, with relu activation function
#         x = F.relu(self.t_conv1(x))
#         x = F.relu(self.t_conv2(x))
#         x = F.relu(self.t_conv3(x))
#         # output layer (with sigmoid for scaling from 0 to 1)
#         x = F.sigmoid(self.t_conv4(x))
#         return x

