import torch
from torch import nn

from src.models.facial_recognition import Backbone


class IDLoss(nn.Module):
    def __init__(self, device, ir_se50_weights: str = '/workspace/saved/models/model_ir_se50.pth'):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(ir_se50_weights))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.facenet = self.facenet.to(device)

        self.device = device
        
        self.loss_values = []
        self.batch_sizes = []

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x.to(self.device))
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, avg: bool = True):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        if avg:
            return loss / count
        
        return loss, count

    def update(self, y_hat, y):
        loss, count = self.forward(y_hat, y, avg=False)
        self.loss_values.append(loss.detach().cpu())
        self.batch_sizes.append(count)

    def compute(self):
        return sum(self.loss_values) / sum(self.batch_sizes)

    def reset(self):
        self.loss_values.clear()
        self.batch_sizes.clear()
