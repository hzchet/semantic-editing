import copy

import torch
import clip
import numpy as np

from src.models.base import BaseInferencer
from src.models.stylegan2 import Generator
from src.models.delta_edit import DeltaMapper
from src.models.styleclip.global_directions.utils import GetDt
from .utils import decoder_validate


class DeltaEditInferencer(BaseInferencer):
    def __init__(
        self,
        stylegan2_ckpt: str,
        relevance_matrix_path: str,
        clip_ckpt: str,
        delta_mapper_ckpt: str,
        device: torch.device,
        stylegan_size: int = 1024,
        style_dim: int = 512,
        threshold: float = 0.03
    ):
        super().__init__()
        self.generator = Generator(
            size=stylegan_size,
            style_dim=style_dim,
            n_mlp=8
        )
        ckpt = torch.load(stylegan2_ckpt)
        self.generator.load_state_dict(ckpt['g_ema'], strict=False)
        self.generator.eval()
        self.generator = self.generator.to(device)
        
        self.fs3 = np.load(relevance_matrix_path)
        np.set_printoptions(suppress=True)
        
        self.clip_model, _ = clip.load(clip_ckpt, device=device)
        
        self.delta_mapper = DeltaMapper()
        ckpt = torch.load(delta_mapper_ckpt)
        self.delta_mapper.load_state_dict(ckpt)
        self.delta_mapper = self.delta_mapper.to(device)
        
        self.device = device
        self.threshold = threshold
    
    @staticmethod
    def _improved_ds(ds, select):
        ds_imp = copy.deepcopy(ds)
        ds_imp[:, select] = 0
        return ds_imp
    
    @torch.inference_mode()
    def __call__(
        self,
        w_latent,
        s_latent,
        c_latent,
        text_prompt,
        neutral_prompt,
        *args,
        **kwargs
    ):
        classnames = [text_prompt, neutral_prompt]
        dt = GetDt(classnames, self.clip_model, normalize=False)
        select = np.dot(self.fs3, dt)
        select = np.abs(select) < self.threshold
        dt = torch.from_numpy(dt)
        dt = dt / dt.norm(dim=-1, keepdim=True).float().clamp(min=1e-5)
        
        clip_feat = torch.cat(
            [c_latent, dt.unsqueeze(0).expand(c_latent.shape[0], -1)], 
            dim=1
        ).to(self.device)
        
        s_latent = s_latent.to(self.device)
        w_latent = w_latent.to(self.device)
        
        fake_delta_s = self.delta_mapper(s_latent, clip_feat)
        improved_fake_delta_s = self._improved_ds(fake_delta_s, select)
        
        return self.to_image(w_latent, s_latent + improved_fake_delta_s), \
               self.to_image(w_latent, s_latent)
        
    def to_image(self, w_latent, s_latent, *args, **kwargs):
        return decoder_validate(self.generator, s_latent, w_latent).detach().cpu()
