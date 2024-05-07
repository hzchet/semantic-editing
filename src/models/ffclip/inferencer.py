import copy

import torch
import clip
import numpy as np

from src.models.base import BaseInferencer
from src.models.stylegan2 import Generator
from src.models.ffclip import DynamicMapper


class FFCLIPInferencer(BaseInferencer):
    def __init__(
        self,
        stylegan2_ckpt: str,
        clip_ckpt: str,
        ffclip_ckpt: str,
        device: torch.device,
        stylegan_size: int = 1024,
        style_dim: int = 512
    ):
        super().__init__()
        self.generator = Generator(
            size=stylegan_size,
            style_dim=style_dim,
            n_mlp=8
        )
        ckpt = torch.load(stylegan2_ckpt)
        self.generator.load_state_dict(ckpt['g_ema'], strict=True)
        self.generator.eval()
        self.generator = self.generator.to(device)
        
        self.clip_model, _ = clip.load(clip_ckpt, device=device)
        
        self.ffclip = DynamicMapper()
        self.ffclip.load_state_dict(torch.load(ffclip_ckpt))
        self.ffclip = self.ffclip.to(device)
        
        self.device = device
    
    @torch.inference_mode()
    def __call__(
        self,
        w_latent,
        text_prompt,
        *args,
        **kwargs
    ):
        tokens = clip.tokenize(text_prompt).to(self.device)
        attribute = self.clip_model.encode_text(tokens).unsqueeze(0)
        
        w_latent = w_latent.to(self.device)
        with torch.autocast('cuda'):
            delta_w = self.ffclip(w_latent, attribute)
        
        return self.to_image(w_latent + delta_w), self.to_image(w_latent)
        
    def to_image(self, w_latent, *args, **kwargs):
        return self.generator([w_latent], input_is_latent=True, randomize_noise=False, truncation=1)[0].detach().cpu()
