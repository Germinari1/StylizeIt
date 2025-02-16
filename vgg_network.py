##########################################################################
# Description: Sets up the convolutional neural network (based on VGG19) used in this project
# Notes:
#    - There is an option to use average pooling instead of max pooling
##########################################################################
from typing import List
from torch import nn, Tensor
from torchvision import models


class Vgg19(nn.Module):
    """
    Sets up the convolutional neural network (based on VGG19) used in this project
    """
    def __init__(self, use_avg_pooling: bool = False):
        super(Vgg19, self).__init__()
        # Load the pretrained VGG19 model
        layers = models.vgg19(pretrained=True).features

        # Option to use average pooling instead of max pooling
        if use_avg_pooling:
            layers = nn.Sequential(
                *[x if not isinstance(x, nn.MaxPool2d) else nn.AvgPool2d(2) for x in layers])

        # Create network blocks
        self.block1 = nn.Sequential()
        self.block2 = nn.Sequential()
        self.block3 = nn.Sequential()
        self.block4 = nn.Sequential()
        self.block5 = nn.Sequential()

        # populate with layers
        for x in range(2):
            self.block1.add_module(str(x), layers[x])
        for x in range(2, 7):
            self.block2.add_module(str(x), layers[x])
        for x in range(7, 12):
            self.block3.add_module(str(x), layers[x])
        for x in range(12, 21):
            self.block4.add_module(str(x), layers[x])
        for x in range(21, 30):
            self.block5.add_module(str(x), layers[x])

        # Normalize the input image
        mean = Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self._mean = nn.Parameter(mean, requires_grad=False)
        self._std = nn.Parameter(std, requires_grad=False)

        for p in self.parameters():
            p.requires_grad = False

        self.eval()

    def forward(self, img: Tensor) -> List[Tensor]:
        # normalize input image
        img = (img - self._mean) / self._std

        # forward propagation
        h_relu1 = self.block1(img)
        h_relu2 = self.block2(h_relu1)
        h_relu3 = self.block3(h_relu2)
        h_relu4 = self.block4(h_relu3)
        h_relu5 = self.block5(h_relu4)

        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
