import torch.nn as nn
import torchvision.models as models

from models.wideresnet import wrn
from models.transformer import vit

class ModelSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, depth=None, width=None):
        super(ModelSimCLR, self).__init__()
        self.model_dict = {"resnet18": lambda: models.resnet18(pretrained=False, num_classes=out_dim),
                           "resnet50": lambda: models.resnet50(pretrained=False, num_classes=out_dim),
                           "wrn": lambda: wrn(depth=depth, widen_factor=width, num_classes=out_dim),
                           "vit": lambda: vit(depth=depth, width=width, num_classes=out_dim),
                           }

        self.backbone = self.model_dict.get(base_model)()


        if base_model in ['wrn']:
            # add mlp projection head
            dim_mlp = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
        elif base_model in ['vit']:
            dim_mlp = self.backbone.mlp_head[1].in_features
            self.backbone.mlp_head[1] = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.mlp_head[1])
            

    def forward(self, x):
        return self.backbone(x)
