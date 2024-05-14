import torch
import torch.nn as nn
import torchvision.transforms as transforms
import clip
import torch.nn.functional as F

from src.models.styleclip.global_directions.utils import imagenet_templates


class NCELoss(nn.Module):
    def __init__(self, device, src_text='face'):
        super().__init__()
        self.device = device
        clip_model, clip_preprocess = clip.load('ViT-B/32')
        self.clip_model = clip_model
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])  
        
        self.src_feat = self.encode_text(src_text)
        
    def encode_image(self, x):
        return self.clip_model.encode_image(
            self.preprocess(x).to(self.device)
        )
    
    def encode_text(self, x):
        text = [t.format(x) for t in imagenet_templates]
        tokens = clip.tokenize(text).to(self.device)
        
        text_features = self.clip_model.encode_text(tokens).detach()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def forward(self, x_aug, latent_c, delta_c):
        latent_aug_c = self.encode_image(x_aug)
        latent_aug_c /= latent_aug_c.clone().norm(dim=-1, keepdim=True)
        
        q = latent_aug_c - latent_c
        q /= q.clone().norm(dim=-1, keepdim=True)
        
        k_neg = self.src_feat - latent_c[:1]
        k_neg /= k_neg.clone().norm(dim=-1, keepdim=True)

        pos_logit = torch.sum(q * delta_c, dim=1, keepdim=True)
        neg_logits = q @ k_neg.transpose(-2, -1)
        logits = torch.cat([pos_logit, neg_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=q.device)
        loss = F.cross_entropy(logits / 0.1, labels, reduction='mean')

        return loss
