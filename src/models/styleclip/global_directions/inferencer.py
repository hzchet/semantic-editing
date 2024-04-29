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
        num_img = 100_000
        self.manipulator.GenerateS(num_img=num_img)
        self.manipulator.GetCodeMS()
        np.set_printoptions(suppress=True)

        self.fs3 = np.load(relevance_matrix_path)

    def __call__(self, w_latents, text_prompt: str, neutral_text: str = 'face'):
        self.manipulator.num_images = w_latents.shape[0]

        s_latents = self.manipulator.G.synthesis.W2S(w_latents)
        s_latents = self.manipulator.S2List(s_latents)

        classnames = [text_prompt, neutral_text]
        dt = GetDt(classnames, self.clip_model)

        boundary, _ = GetBoundary(self.fs3, dt, self.manipulator,
                                  threshold=self.beta)
        codes = self.manipulator.MSCode(s_latents, boundary)
        out = self.manipulator.GenerateImg(codes)

        return out.squeeze(1)
