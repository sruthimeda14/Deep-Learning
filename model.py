import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.maxpool(self.relu(self.batchnorm(self.conv(x))))
        return x

class BallDetection(nn.Module):
    def __init__(self, in_ch, dropout_p, num_frames_sequence):
        super(BallDetection, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1)
        self.batchnorm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # Convolution Blocks
        self.convblock1 = ConvBlock(in_channels=64, out_channels=64)
        self.convblock2 = ConvBlock(in_channels=64, out_channels=64)
        self.convblock3 = ConvBlock(in_channels=64, out_channels=128)
        self.convblock4 = ConvBlock(in_channels=128, out_channels=128)
        self.convblock5 = ConvBlock(in_channels=128, out_channels=256)
        self.convblock6 = ConvBlock(in_channels=256, out_channels=256)

        # Fully Connected Layers
        self.fc1 = nn.Linear(in_features=2560, out_features=1792)
        self.fc2 = nn.Linear(in_features=1792, out_features=896)
        self.fc3 = nn.Linear(in_features=896, out_features=448)
        self.fc4 = nn.Linear(in_features=448, out_features=256)
        self.fc5 = nn.Linear(in_features=256, out_features=128)
        self.fc6 = nn.Linear(in_features=128, out_features=64)
        self.fc7 = nn.Linear(in_features=64, out_features=32)
        self.fc8 = nn.Linear(in_features=32, out_features=16)
        self.fc9 = nn.Linear(in_features=16, out_features=8)
        self.fc10 = nn.Linear(in_features=8, out_features=2)

        self.dropout2d = nn.Dropout2d(p=dropout_p)
        self.dropout1d = nn.Dropout(p=dropout_p)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.batchnorm(self.conv1(x)))
        x = self.convblock2(self.convblock1(x))
        x = self.dropout2d(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.dropout2d(x)
        x = self.convblock5(x)
        features = self.convblock6(x)
        x = self.dropout2d(features)
        x = x.contiguous().view(x.size(0), -1)
        x = self.dropout1d(self.relu(self.fc1(x)))
        x = self.dropout1d(self.relu(self.fc2(x)))
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.fc9(x)
        x = self.fc10(x)

        return x


class CombinedModel(nn.Module):
    def __init__(self, in_ch, dropout_p, num_frames_sequence, local=False):
        super(CombinedModel, self).__init__()
        self.local = local
        # Create two instances of BallDetection
        self.model1 = BallDetection(in_ch, dropout_p,num_frames_sequence)
        if local:
            self.model2 = BallDetection(in_ch, dropout_p, num_frames_sequence)

    def forward(self, x):
        # Forward pass for the first input through the first model
        out1 = self.model1(x)
        # if self.local:
        #     # Forward pass for the second input through the second model
        #     out2 = self.model2(x2)

        # # Return the outputs of both models
        # return out1, out2
        return out1