import torch.nn as nn
from collections import OrderedDict
from coral_pytorch.layers import CoralLayer
import timm
import torch
import volo
from volo import volo_d1


class VoloBackbone(nn.Module):
    def __init__(
        self,
        base_name: str = "volo_d1",
        pretrained=False,
    ):
        """Initialize"""
        self.base_name = base_name
        super().__init__()

        base_model = volo.volo_d1(pretrained=pretrained, return_dense=False)
        base_model.reset_classifier(num_classes=0)
        self.backbone = base_model
        self.output_dim = 384

    def features(self, x):
        x_cls = self.forward(x)
        return x_cls

    def forward(self, x):
        """Forward"""
        h = self.backbone(x)
        return h


def build_layers(n, method, prev_features, current_features, down=True):
    layers = OrderedDict()
    for i in range(n):
        layers[str(i)] = method(prev_features, current_features)
        prev_features = current_features
        if down:
            current_features = current_features // 2
        else:
            current_features = current_features * 2
    return layers, prev_features


class DeepNormal(nn.Module):
    def __init__(
        self,
        cnn_backbone=None,
        input_dim=1,
        lin_start=2048,
        n_lin=6,
    ):
        super().__init__()

        self.cnn_backbone = cnn_backbone

        previous_feature_size = input_dim
        current_feature_size = lin_start
        last_feature_size = input_dim
        self.blocks = None
        if n_lin > 0:
            lin_layers, last_feature_size = build_layers(
                n_lin, self.block, previous_feature_size, current_feature_size
            )
        self.blocks = nn.Sequential(lin_layers)

        # Mean parameters
        self.mean_layer = nn.Sequential(
            nn.Linear(last_feature_size, last_feature_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(last_feature_size, 1),
            nn.Sigmoid(),
        )

        # Standard deviation parameters
        self.std_layer = nn.Sequential(
            nn.Linear(last_feature_size, last_feature_size),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(last_feature_size, 1),
            nn.Sigmoid(),  # enforces positivity
        )

    def block(self, in_features, out_features):
        lin = nn.Linear(in_features, out_features)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(lin, relu)

    def forward(self, x):
        # Shared embedding
        shared = self.cnn_backbone(x)
        shared = self.blocks(shared)

        # Parametrization of the mean
        u = self.mean_layer(shared)

        # Parametrization of the standard deviation
        q = self.std_layer(shared)

        return torch.concat([u, q], axis=-1)


class CoralModel(torch.nn.Module):
    def __init__(self, backbone, input_dim, num_classes):
        super(CoralModel, self).__init__()

        self.backbone_cnn = backbone

        ### Specify CORAL layer
        self.fc = CoralLayer(size_in=input_dim, num_classes=num_classes)
        ###--------------------------------------------------------------------###

    def forward(self, x):
        x = self.backbone_cnn(x)
        x = x.view(x.size(0), -1)  # flatten

        ##### Use CORAL layer #####
        logits = self.fc(x)
        probas = torch.sigmoid(logits)
        ###--------------------------------------------------------------------###

        return logits, probas


class RegressionModel(nn.Module):
    def __init__(
        self,
        cnn_backbone=None,
        input_dim=1,
        output_dim=1,
        lin_start=2048,
        n_lin=6,
        sigmoid=True,
    ):
        super(RegressionModel, self).__init__()
        self.cnn_backbone = cnn_backbone

        ############ Dense layers ########
        previous_feature_size = input_dim
        current_feature_size = lin_start
        last_feature_size = input_dim
        self.blocks = None
        if n_lin > 0:
            lin_layers, last_feature_size = build_layers(
                n_lin, self.block, previous_feature_size, current_feature_size
            )
            self.blocks = nn.Sequential(lin_layers)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(last_feature_size, output_dim)
        self.activation = nn.Sigmoid()
        self.sigmoid = sigmoid

    def block(self, in_features, out_features):
        lin = nn.Linear(in_features, out_features)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(lin, relu)

    def forward(self, x):
        z = x

        if self.cnn_backbone:
            z = self.cnn_backbone.features(z)

        if self.blocks:
            z = self.blocks(z)

        z = self.linear(z)

        if self.output_dim == 1:
            z = torch.reshape(z, (z.shape[0],))

        if self.sigmoid:
            z = self.activation(z)
        return z


class FeatureClassificationModel(nn.Module):
    def __init__(
        self,
        cnn_backbone,
        input_dim=1024,
        n_input=2,
        n_classes=2,
        lin_start=2048,
        n_lin=6,
    ):
        super(FeatureClassificationModel, self).__init__()
        self.cnn_backbone = cnn_backbone
        self.blocks = None

        previous_feature_size = n_input * input_dim
        current_feature_size = lin_start
        last_feature_size = previous_feature_size
        lin_layers, last_feature_size = build_layers(
            n_lin, self.block, previous_feature_size, current_feature_size
        )
        self.blocks = nn.Sequential(lin_layers)
        self.age_gap = nn.Linear(last_feature_size, n_classes)

    def block(self, in_features, out_features):
        lin = nn.Linear(in_features, out_features)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(lin, relu)

    def get_z(self, x):
        z = self.cnn_backbone.features(x)
        return z

    def forward(self, x):
        x1, x2 = x
        z1 = self.get_z(x1)
        z2 = self.get_z(x2)
        z = torch.cat([z1, z2], axis=-1)
        if self.blocks:
            z = self.blocks(z)
        z = self.age_gap(z)
        return z


import os
import torch
from torch import nn
from torch.nn import functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )  # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001,  # value found in tensorflow
            momentum=0.1,  # default pytorch value
            affine=True,
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block35(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )

        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block17(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(128, 128, kernel_size=(7, 1), stride=1, padding=(3, 0)),
        )

        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block8(nn.Module):
    def __init__(self, scale=1.0, noReLU=False):
        super().__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(192, 192, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        )

        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_6a(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2),
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Mixed_7a(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2),
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2),
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2),
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionResnetV1(nn.Module):
    """Inception Resnet V1 model with optional loading of pretrained weights.

    Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
    datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
    requested and cached in the torch cache. Subsequent instantiations use the cache rather than
    redownloading.

    Keyword Arguments:
        pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
            (default: {None})
        classify {bool} -- Whether the model should output classification probabilities or feature
            embeddings. (default: {False})
        num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
            equal to that used for the pretrained model, the final linear layer will be randomly
            initialized. (default: {None})
        dropout_prob {float} -- Dropout probability. (default: {0.6})
    """

    def __init__(self, dropout_prob=0.6, device=None, output_dim=1024, normalize=True):
        super().__init__()

        # Define layers
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(1792, output_dim, bias=False)
        self.last_bn = nn.BatchNorm1d(output_dim, eps=0.001, momentum=0.1, affine=True)

        self.device = torch.device("cpu")
        if device is not None:
            self.device = device
            self.to(device)

    def features(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        return x

    def forward(self, x):
        """Calculate embeddings or logits given a batch of input image tensors.

        Arguments:
            x {torch.tensor} -- Batch of image tensors representing faces.

        Returns:
            torch.tensor -- Batch of embedding vectors or multinomial logits.
        """
        x = self.features(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x

    def triplet_forward(self, x1, x2, x3):
        return self.forward(x1), self.forward(x2), self.forward(x3)


class VGGFace(nn.Module):
    def __init__(self, start_filters=64, output_dim=2622):
        super().__init__()
        self.model = nn.Sequential(
            self.conv_block(3, start_filters, kernel_size=3, zero_pad=1),
            self.conv_block(start_filters, start_filters, kernel_size=3, zero_pad=1),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            self.conv_block(
                start_filters, start_filters * 2, kernel_size=3, zero_pad=1
            ),
            self.conv_block(
                start_filters * 2, start_filters * 2, kernel_size=3, zero_pad=1
            ),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            self.conv_block(
                start_filters * 2, start_filters * 4, kernel_size=3, zero_pad=1
            ),
            self.conv_block(
                start_filters * 4, start_filters * 4, kernel_size=3, zero_pad=1
            ),
            self.conv_block(
                start_filters * 4, start_filters * 4, kernel_size=3, zero_pad=1
            ),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            self.conv_block(
                start_filters * 4, start_filters * 8, kernel_size=3, zero_pad=1
            ),
            self.conv_block(
                start_filters * 8, start_filters * 8, kernel_size=3, zero_pad=1
            ),
            self.conv_block(
                start_filters * 8, start_filters * 8, kernel_size=3, zero_pad=1
            ),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            self.conv_block(
                start_filters * 8, start_filters * 8, kernel_size=3, zero_pad=1
            ),
            self.conv_block(
                start_filters * 8, start_filters * 8, kernel_size=3, zero_pad=1
            ),
            self.conv_block(
                start_filters * 8, start_filters * 8, kernel_size=3, zero_pad=1
            ),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            self.conv_block(start_filters * 8, start_filters * 64, kernel_size=7),
            nn.Dropout(0.5),
            self.conv_block(start_filters * 64, start_filters * 64, kernel_size=1),
            nn.Dropout(0.5),
            nn.Conv2d(start_filters * 64, output_dim, kernel_size=1),
        )

        self.output_dim = output_dim

    def conv_block(self, input_dim, output_dim, kernel_size, zero_pad=0):
        conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(output_dim)
        if zero_pad > 0:
            return nn.Sequential(conv, relu, bn, nn.ZeroPad2d(zero_pad))
        return nn.Sequential(conv, relu, bn)

    def features(self, x):
        x = self.model(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        return x
