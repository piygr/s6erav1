import torch.nn as nn
import torch.nn.functional as F


class OptimizedNet(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(OptimizedNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), #28x28x
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1), #14x14x16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),    #14x14x16
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3),    #5x5x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3),  # 5x5x32
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv_linear = nn.Conv2d(32, 10, 1)


    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = self.conv3(x)

        x = self.conv_linear(x)

        x = F.avg_pool2d(x, 4)

        x = x.view(x.size(0), 10)
        return F.log_softmax(x, dim=1)
