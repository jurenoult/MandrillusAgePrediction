import torch.nn as nn
from collections import OrderedDict
from coral_pytorch.layers import CoralLayer
import torch
import volo
from volo.utils import load_pretrained_weights
import torchfile
from torch.nn.init import trunc_normal_
import timm


class boundReLU(nn.Module):
    def __init__(self, min_value, max_value):
        super(boundReLU, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        return torch.min(torch.max(self.min_value, x), self.max_value)


class SequentialModel(nn.Module):
    def __init__(self, backbone, head):
        super(SequentialModel, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


class VoloBackbone(nn.Module):
    def __init__(
        self,
        base_name: str = "volo_d1",
    ):
        """Initialize"""
        self.base_name = base_name
        super().__init__()

        if base_name == "volo_d1":
            base_model = timm.create_model("volo_d1_224")
            self.output_dim = 384 * 197
        if base_name == "volo_d2":
            base_model = timm.create_model("volo_d2_224")
            self.output_dim = 512 * 197
        if base_name == "volo_d3":
            base_model = timm.create_model("volo_d3_224")
            self.output_dim = 512 * 197
        if base_name == "volo_d4":
            base_model = timm.create_model("volo_d4_224")
            self.output_dim = 768 * 197
        if base_name == "volo_d5":
            base_model = timm.create_model("volo_d5_224")
            self.output_dim = 768 * 197

        # base_model.reset_classifier(num_classes=0)
        self.backbone = base_model

    def load_weights(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        self.backbone.load_state_dict(checkpoint)
        self.backbone.return_dense = False
        self.backbone.reset_classifier(num_classes=0)

    def features(self, x):
        x = self.backbone.forward_features(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        """Forward"""
        x = self.features(x)
        return x


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


class RegressionHead(nn.Module):
    def __init__(
        self,
        input_dim=1,
        output_dim=1,
        lin_start=2048,
        n_lin=6,
        sigmoid=True,
    ):
        super(RegressionHead, self).__init__()

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
        self.linear = nn.Linear(last_feature_size, output_dim, bias=False)

        if sigmoid:
            self.activation = nn.Sigmoid()
        else:
            self.activation = None  # nn.ReLU()
        self.sigmoid = sigmoid

    def block(self, in_features, out_features):
        lin = nn.Linear(in_features, out_features)
        gelu = nn.ReLU()
        return nn.Sequential(lin, gelu)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=0.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.blocks:
            x = self.blocks(x)
        x = self.linear(x)

        if self.output_dim == 1:
            x = torch.reshape(x, (x.shape[0],))

        if self.activation:
            x = self.activation(x)
        return x


class DinoV2(nn.Module):
    def __init__(self, dino_type: str):
        super(DinoV2, self).__init__()
        self.output_dim = 384
        if dino_type == "small":
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        if dino_type == "medium":
            self.output_dim = 768
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        if dino_type == "large":
            self.output_dim = 1000
            self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_lc")

    def features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.features(x)


class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim=1024,
        n_input=2,
        n_classes=2,
        lin_start=256,
        n_lin=3,
        use_concat=True,
    ):
        super(ClassificationHead, self).__init__()
        self.blocks = None
        self.use_concat = use_concat

        previous_feature_size = n_input * input_dim
        # previous_feature_size = input_dim
        current_feature_size = lin_start
        last_feature_size = previous_feature_size
        if n_lin > 0:
            lin_layers, last_feature_size = build_layers(
                n_lin, self.block, previous_feature_size, current_feature_size
            )
            self.blocks = nn.Sequential(lin_layers)
        self.class_layer = nn.Linear(last_feature_size, n_classes, bias=False)

    def block(self, in_features, out_features):
        lin = nn.Linear(in_features, out_features)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(lin, relu)

    def get_z(self, x):
        z = self.backbone.features(x)
        return z

    def forward(self, x):
        x1, x2 = x
        z1 = self.get_z(x1)
        z2 = self.get_z(x2)

        if self.use_concat:
            z = torch.cat([z1, z2], axis=-1)
        else:
            z = torch.sub(z1, z2)

        if self.blocks:
            z = self.blocks(z)
        z = self.class_layer(z)
        return z


class VGGFace(nn.Module):
    def __init__(self, start_filters=64, output_dim=2622):
        super().__init__()

        self.start_filters = start_filters
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, start_filters, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(start_filters, start_filters, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(start_filters, start_filters * 2, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(start_filters * 2, start_filters * 2, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(start_filters * 2, start_filters * 4, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(start_filters * 4, start_filters * 4, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(start_filters * 4, start_filters * 4, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(start_filters * 4, start_filters * 8, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(start_filters * 8, start_filters * 8, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(start_filters * 8, start_filters * 8, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(start_filters * 8, start_filters * 8, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(start_filters * 8, start_filters * 8, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(start_filters * 8, start_filters * 8, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(start_filters * 8 * 7 * 7, start_filters * 64)
        self.fc7 = nn.Linear(start_filters * 64, start_filters * 64)
        self.fc8 = nn.Linear(start_filters * 64, output_dim)
        self.create_bn()

        self.last_output_dim = output_dim
        self.output_dim = start_filters * 8 * 7 * 7

    def load_weights(self, path="data/pretrained/VGG_FACE.t7"):
        """Function to load luatorch pretrained

        from: https://www.robots.ox.ac.uk/~vgg/software/vgg_face/

        Args:
            path: path for the luatorch pretrained
        """
        assert (
            self.start_filters == 64 and self.last_output_dim == 2622
        ), "You must use the correct model size to load the pretrained weight."
        model = torchfile.load(path)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(
                        self_layer.weight
                    )[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[
                        ...
                    ]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(
                        self_layer.weight
                    )[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[
                        ...
                    ]
        print(f"Loaded model from pretrained model at path : {path}")

    def create_bn(self):
        self.bn_1_1 = nn.BatchNorm2d(self.start_filters)
        self.bn_1_2 = nn.BatchNorm2d(self.start_filters)
        self.bn_2_1 = nn.BatchNorm2d(self.start_filters * 2)
        self.bn_2_2 = nn.BatchNorm2d(self.start_filters * 2)
        self.bn_3_1 = nn.BatchNorm2d(self.start_filters * 4)
        self.bn_3_2 = nn.BatchNorm2d(self.start_filters * 4)
        self.bn_3_3 = nn.BatchNorm2d(self.start_filters * 4)
        self.bn_4_1 = nn.BatchNorm2d(self.start_filters * 8)
        self.bn_4_2 = nn.BatchNorm2d(self.start_filters * 8)
        self.bn_4_3 = nn.BatchNorm2d(self.start_filters * 8)
        self.bn_5_1 = nn.BatchNorm2d(self.start_filters * 8)
        self.bn_5_2 = nn.BatchNorm2d(self.start_filters * 8)
        self.bn_5_3 = nn.BatchNorm2d(self.start_filters * 8)
        self.bn_6 = nn.BatchNorm1d(self.start_filters * 64)
        self.bn_7 = nn.BatchNorm1d(self.start_filters * 64)

    def features(self, x):
        x = self.bn_1_1(F.relu(self.conv_1_1(x)))
        x = self.bn_1_2(F.relu(self.conv_1_2(x)))
        x = F.max_pool2d(x, 2, 2)  # 224 -> 112
        x = self.bn_2_1(F.relu(self.conv_2_1(x)))
        x = self.bn_2_2(F.relu(self.conv_2_2(x)))
        x = F.max_pool2d(x, 2, 2)  # 112 -> 56
        x = self.bn_3_1(F.relu(self.conv_3_1(x)))
        x = self.bn_3_2(F.relu(self.conv_3_2(x)))
        x = self.bn_3_3(F.relu(self.conv_3_3(x)))
        x = F.max_pool2d(x, 2, 2)  # 56 -> 28
        x = self.bn_4_1(F.relu(self.conv_4_1(x)))
        x = self.bn_4_2(F.relu(self.conv_4_2(x)))
        x = self.bn_4_3(F.relu(self.conv_4_3(x)))
        x = F.max_pool2d(x, 2, 2)  # 28 -> 14
        x = self.bn_5_1(F.relu(self.conv_5_1(x)))
        x = self.bn_5_2(F.relu(self.conv_5_2(x)))
        x = self.bn_5_3(F.relu(self.conv_5_3(x)))
        x = F.max_pool2d(x, 2, 2)  # 14 -> 7
        x = x.view(
            x.size(0), -1
        )  # 7x7x512 => This part is used as features for age regression in deepface
        # x = self.bn_6(F.relu(self.fc6(x)))
        # x = F.dropout(x, 0.5, self.training)
        # x = self.bn_7(F.relu(self.fc7(x)))
        # x = F.dropout(x, 0.5, self.training)
        # x = self.fc8(x)

        # x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        return x
