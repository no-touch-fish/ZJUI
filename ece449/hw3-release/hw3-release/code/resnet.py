import torch.nn as nn


class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and is also
                          the number of channels of conv layers of Block.
        """
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.batchnorm1 = nn.BatchNorm2d(num_features=num_channels)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.batchnorm2 = nn.BatchNorm2d(num_features=num_channels)
        self.relu = nn.ReLU()


    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """
        res = x
        y = self.batchnorm1(self.conv1(x))
        y = self.relu(y)
        y = self.batchnorm2(self.conv2(y))
        y = self.relu(res+y)
        return y


class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=num_channels,kernel_size=3,stride=2,padding=1,bias=False)
        self.batchnorm1 = nn.BatchNorm2d(num_features=num_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.block = Block(num_channels=num_channels)
        self.adapt = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(in_features=num_channels,out_features=num_classes)

    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.block(x)
        x = self.adapt(x)
        # print('after adapt',x.shape)
        x = x.view(x.size(0), -1)
        y = self.linear(x)
        return y