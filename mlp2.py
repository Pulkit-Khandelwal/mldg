import math

import torch.nn as nn

from ops import *

## Load the model
# model_conv = models.alexnet(pretrained='imagenet')
#
# freeze_layers = True
# n_class = 3
#
# for i, param in model_conv.named_parameters():
#     print(i, type(i), type(param))
#
# print('enddddd')
#
# #freeze all the layers except the classifier on the top
# if freeze_layers:
#   for i, param in model_conv.features.named_parameters():
#       param.requires_grad = False


#To view which layers are freeze and which layers are not freezed:
# for name, params in model_conv.named_parameters():
#     print(name, params.requires_grad)


from torchvision import datasets, models, transforms
from torchvision.models.alexnet import AlexNet
import torch.utils.model_zoo as model_zoo

#### Monkey Patch the forward function of the
#### AlexNet class

# when you add the convolution and batch norm, below will be useful

# def __init__(self):
#     super(AlexNet, self).__init__()
#     for m in self.modules():
#         if isinstance(m, nn.Conv2d):
#             n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             m.weight.data.normal_(0, math.sqrt(2. / n))
#         elif isinstance(m, nn.BatchNorm2d):
#             m.weight.data.fill_(1)
#             m.bias.data.zero_()

#AlexNet.__init__ = __init__


def yo(self, x, meta_loss=None, num_classes=1000, meta_step_size=None, stop_gradient=False):
    #super(AlexNet).__init__()
    self.params = list(AlexNet.parameters())

    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    # Features Block

    # Block 1
    x = conv2d(inputs=x, weight=self.params[0], bias=self.params[1], meta_step_size=0.001,
           stride=4, padding=2, dilation=1, groups=1, meta_loss=None,
           stop_gradient=False)

    x = F.relu(inputs=x, inplace=True)

    x = maxpool(inputs=x, kernel_size=3, stride=2)


    # Block 2
    x = conv2d(inputs=x, weight=self.params[2], bias=self.params[3], meta_step_size=0.001,
           padding=2, dilation=1, groups=1, meta_loss=None,
           stop_gradient=False)

    x = F.relu(inputs=x, inplace=True)

    x = maxpool(inputs=x, kernel_size=3, stride=2)

    # Block 3
    x = conv2d(inputs=x, weight=self.params[4], bias=self.params[5], meta_step_size=0.001,
           padding=1, dilation=1, groups=1, meta_loss=None,
           stop_gradient=False)

    x = F.relu(inputs=x, inplace=True)

    # Block 4
    x = conv2d(inputs=x, weight=self.params[6], bias=self.params[7], meta_step_size=0.001,
           padding=1, dilation=1, groups=1, meta_loss=None,
           stop_gradient=False)

    x = F.relu(inputs=x, inplace=True)

    # Block 5
    x = conv2d(inputs=x, weight=self.params[8], bias=self.params[9], meta_step_size=0.001,
           padding=2, dilation=1, groups=1, meta_loss=None,
           stop_gradient=False)

    x = F.relu(inputs=x, inplace=True)

    x = maxpool(inputs=x, kernel_size=3, stride=2)

    # Classifier Block

    x = dropout(inputs=x)

    # Block 1
    x = linear(inputs=x,
               weight=self.params[10], bias=self.params[11],
               meta_loss=meta_loss,
               meta_step_size=meta_step_size,
               stop_gradient=stop_gradient)

    x = F.relu(inputs=x, inplace=True)

    x = dropout(inputs=x)

    # Block 2
    x = linear(inputs=x,
               weight=self.params[12], bias=self.params[13],
               meta_loss=meta_loss,
               meta_step_size=meta_step_size,
               stop_gradient=stop_gradient)

    x = F.relu(inputs=x, inplace=True)

    self.last_x = linear(inputs=x,
               weight=self.params[14], bias=self.params[15],
               meta_loss=meta_loss,
               meta_step_size=meta_step_size,
               stop_gradient=stop_gradient)

    end_points = {'Predictions': F.softmax(input=self.last_x, dim=-1)}

    return self.last_x, end_points

# class MLP(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(512, 512)
#         self.fc2 = nn.Linear(512, num_classes)
#
#         # when you add the convolution and batch norm, below will be useful
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def forward(self, x, meta_loss=None, meta_step_size=None, stop_gradient=False):
#
#         x = linear(inputs=x,
#                    weight=self.fc1.weight,
#                    bias=self.fc1.bias,
#                    meta_loss=meta_loss,
#                    meta_step_size=meta_step_size,
#                    stop_gradient=stop_gradient)
#
#         x = F.relu(inputs=x, inplace=True)
#
#         x = linear(inputs=x,
#                    weight=self.fc2.weight,
#                    bias=self.fc2.bias,
#                    meta_loss=meta_loss,
#                    meta_step_size=meta_step_size,
#                    stop_gradient=stop_gradient)
#
#         end_points = {'Predictions': F.softmax(input=x, dim=-1)}
#
#         return x, end_points


def Net(**kwargs):
    model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    }

    md = AlexNet()
    print(md.forward)
    md.forward = yo
    print(md.forward(self=2, x=None))
    md.load_state_dict(model_zoo.load_url(model_urls['alexnet']))

    md.last_x = linear(inputs=md.params[14].in_features,
                       weight=md.params[14], bias=7,
                       meta_loss=meta_loss,
                       meta_step_size=meta_step_size,
                       stop_gradient=stop_gradient)


    freeze_layers = True
    # freeze the pre-trained features
    if freeze_layers:
        for i, param in md.features.named_parameters():
            param.requires_grad = False
    return md

# # Call the model and check the
# # layers which are frozen
model = Net()
# print(model)
# for name, params in model.named_parameters():
#     print(name, params.requires_grad)
# #
# import h5py
# import numpy as np
# file_path = '/Users/pulkit/Desktop/MLDG-master/data/sketch_train_features.hdf5'
# with h5py.File(file_path, 'r') as f:
#     images = np.array(f['images'])
#     labels = np.array(f['labels'])
#
# print(np.shape(images), np.shape(labels))
#
# #
# # model_resent = models.resnet18(pretrained=True)
# # print(model_resent)