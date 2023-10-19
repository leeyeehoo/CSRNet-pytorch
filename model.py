import collections
import torch.nn as nn
from torchvision import models

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            self._initialize_weights_from_vgg()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        """
        Initialize the weights of the model
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _initialize_weights_from_vgg(self):
        """
        Initialize weights using pretrained VGG16 model
        """
        mod = models.vgg16(pretrained=True)
        self._initialize_weights()

        fsd = collections.OrderedDict()
        frontend_dict_items = list(self.frontend.state_dict().items())
        mod_dict_items = list(mod.state_dict().items())
        for i in range(len(frontend_dict_items)):
            fsd[frontend_dict_items[i][0]] = mod_dict_items[i][1]

        self.frontend.load_state_dict(fsd)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    """
    Construct a layer for the given configuration
    """
    d_rate = 2 if dilation else 1
    layers = []

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    return nn.Sequential(*layers)
