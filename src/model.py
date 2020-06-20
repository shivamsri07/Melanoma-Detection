import torch
import torch.nn as nn
import pretrainedmodels
import torch.nn.functional as F

class SE_Resnext50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(SE_Resnext50_32x4d, self).__init__()

        self.base_model = pretrainedmodels.__dict__["se_resnext50_32x4d"](pretrained="imagenet")

        self.l0 = nn.Linear(2048, 1)

    def forward(self, image, targets):
        bs, _, _, _ = image.shape

        x = self.base_model.features(image)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)

        op = self.l0(x)
        loss = nn.BCEWithLogitsLoss()(op, targets.view(-1, 1).type_as(op))

        return op, loss