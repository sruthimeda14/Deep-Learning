import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import zipfile

def conv2D_output_size(img_size, padding, kernel_size, stride):
    return np.floor(
        (img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0]
        + 1
    ).astype(int), np.floor(
        (img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1]
        + 1
    ).astype(
        int
    )


class EncoderCNN(nn.Module):
    def __init__(self, img_x=90, img_y=120, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        super(EncoderCNN, self).__init__()

        self.img_x = img_x
        self.img_y = img_y
        self.CNN_embed_dim = CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 32, 64, 128, 256
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # conv2D output shapes
        self.conv1_outshape = conv2D_output_size((self.img_x, self.img_y), self.pd1, self.k1, self.s1)  # Conv1 output shape
        self.conv2_outshape = conv2D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv3_outshape = conv2D_output_size(self.conv2_outshape, self.pd3, self.k3, self.s3)
        self.conv4_outshape = conv2D_output_size(self.conv3_outshape, self.pd4, self.k4, self.s4)
        # fully connected layer hidden nodes
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1, padding=self.pd1),
            nn.BatchNorm2d(self.ch1, momentum=0.01),
            nn.ReLU(inplace=True),                      
            # nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2, padding=self.pd2),
            nn.BatchNorm2d(self.ch2, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch2, out_channels=self.ch3, kernel_size=self.k3, stride=self.s3, padding=self.pd3),
            nn.BatchNorm2d(self.ch3, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.ch3, out_channels=self.ch4, kernel_size=self.k4, stride=self.s4, padding=self.pd4),
            nn.BatchNorm2d(self.ch4, momentum=0.01),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2),
        )

        self.drop = nn.Dropout2d(self.drop_p)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(self.ch4 * self.conv4_outshape[0] * self.conv4_outshape[1], self.fc_hidden1)   # fully connected layer, output k classes
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.fc3 = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)   # output = CNN embedding latent variables

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # CNNs
            x = self.conv1(x_3d[:, t, :, :, :])
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.view(x.size(0), -1)           # flatten the output of conv

            # FC layers
            x = F.relu(self.fc1(x))
            # x = F.dropout(x, p=self.drop_p, training=self.training)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)
            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=50,n_frames=9):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.n_frames = n_frames

        # self.LSTM = nn.LSTM(
        #     input_size=self.RNN_input_size,
        #     hidden_size=self.h_RNN,        
        #     num_layers=h_RNN_layers,       
        #     batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        # )
        # self.rnn = nn.rnn(
          
        # )
        self.rnn = nn.RNN(
            self.RNN_input_size,
            self.h_RNN,
            self.h_RNN_layers,
            batch_first = True
        )
        # print(self.RNN_input_size)
        # print(self.h_RNN, self.h_FC_dim)
        self.fc1 = nn.Linear(self.RNN_input_size*self.n_frames,self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    # def forward(self, x_RNN):
        
    #     self.LSTM.flatten_parameters()
    #     RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)  
    #     """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """ 
    #     """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

    #     # FC layers
    #     x = self.fc1(RNN_out[:, -1, :])   # choose RNN_out at the last time step
    #     x = F.relu(x)
    #     x = F.dropout(x, p=self.drop_p, training=self.training)
    #     x = self.fc2(x)

    #     return x

    def forward(self,x):
        use_cuda = torch.cuda.is_available()                   # check if GPU exists
        device = torch.device("cuda" if use_cuda else "cpu") 
        h0 = torch.zeros(self.h_RNN_layers, x.size(0),self.h_RNN).to(device)

        out,_ = self.rnn(x,h0)
        out = out.reshape(out.shape[0],-1)
        out = self.fc1(out)
        return self.fc2(out)






