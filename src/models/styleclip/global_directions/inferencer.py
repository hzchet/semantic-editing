import torch
import clip
import numpy as np

from src.models.base import BaseInferencer
from src.models.styleclip.global_directions.manipulate import Manipulator
from src.models.styleclip.global_directions.utils import GetBoundary, GetDt


class StyleCLIPInferencer(BaseInferencer):
    def __init__(
        self,
        stylegan2_path: str,
        relevance_matrix_path: str,
        clip_ckpt: str,
        device: torch.device,
        alpha: float = 1,
        beta: float = 0.1
    ):
        super().__init__()
        self.device = device
        self.manipulator = Manipulator(device=device)
        self.manipulator.alpha = [alpha]
        self.beta = beta

        self.clip_model, _ = clip.load(clip_ckpt, device=device, jit=False)

        self.manipulator.G = self.manipulator.LoadModel(stylegan2_path, device)
        self.manipulator.SetGParameters()
        num_img = 100
        self.manipulator.GenerateS(num_img=num_img)
        self.manipulator.GetCodeMS()
        np.set_printoptions(suppress=True)

        self.fs3 = np.load(relevance_matrix_path)

    def __call__(
        self,
        w_latent,
        text_prompt: str,
        neutral_text: str = 'face',
        *args,
        **kwargs
    ):
        self.manipulator.num_images = w_latent.shape[0]
        s_latents = self.manipulator.G.synthesis.W2S(w_latent.to(self.device))

        classnames = [text_prompt, neutral_text]
        dt = GetDt(classnames, self.clip_model)

        boundary, _ = GetBoundary(self.fs3, dt, self.manipulator,
                                  threshold=self.beta)
        
        codes = self.manipulator.MSCode(s_latents, boundary)
        out = self.manipulator.GenerateImg(codes)

        return out, self.to_image(s_latents)

    def to_image(self, s_latent, *args, **kwargs):
        return self.manipulator.GenerateImg(s_latent)
