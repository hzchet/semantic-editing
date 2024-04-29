import torch
import clip
import numpy as np

from src.models.base import BaseInferencer
from src.models.styleclip.global_directions.manipulate import Manipulator


class StyleCLIPInferencer(BaseInferencer):
    def __init__(
        self,
        stylegan2_path: str,
        relevance_matrix_path: str,
        clip_ckpt: str,
        device: torch.device
    ):
        super().__init__()
        self.device = device
        self.manipulator = Manipulator(device=device)
        
        self.clip_model, _ = clip.load(clip_ckpt, device=device, jit=False)

        self.manipulator.G = self.manipulator.LoadModel(stylegan2_path, device)
        self.manipulator.SetGParameters()
        num_img = 100_000
        self.manipulator.GenerateS(num_img=num_img)
        self.manipulator.GetCodeMS()
        np.set_printoptions(suppress=True)

        self.fs3 = np.load(relevance_matrix_path)
    
    def __call__(self, images, text_prompt: str, neutral_text: str = 'face'):
        
    