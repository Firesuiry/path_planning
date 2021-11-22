import torch.nn as nn

class ConvFCNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(8, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu3 = nn.Sigmoid()
        self.conv4 = nn.Conv2d(64, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(in_features=2304*8, out_features=512, bias=True)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=512, out_features=128, bias=True)
        self.relu5 = nn.Sigmoid()
        self.fc3 = nn.Linear(in_features=128, out_features=28, bias=True)

    def forward(self, x, torch=None):
        x1 = self.conv1(x)
        x2 = self.relu1(x1)
        x3 = self.conv2(x2)
        x4 = self.relu2(x3)
        x5 = self.conv3(x4)
        x6 = self.relu3(x5)
        x7 = self.conv4(x6)
        x8 = self.relu3(x7)
        x9 = x8.reshape(x8.shape[0], -1)
        x10 = self.fc1(x9)
        x11 = self.relu5(x10)
        x12 = self.fc2(x11)
        x13 = self.relu5(x12)
        x14 = self.fc3(x13)

        resnet_out = x14.reshape(-1, 7, 4)
        action = torch.softmax(resnet_out, -1)
        return action