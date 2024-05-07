from typing import List

import torch
import wandb
from tqdm import tqdm
from torcheval.metrics import FrechetInceptionDistance
import torchvision.utils as vutils
import torch.nn.functional as F

from src.models.base import BaseInferencer
from src.losses import IDLoss, CLIPScore


class Evaluator:
    def __init__(
        self,
        inferencer: BaseInferencer,
        loader,
        device,
        prompts: List[str],
        neutral_prompts: List[str] = None,
        wandb_project: str = 'semantic-editing',
        wandb_run_name: str = f'StyleCLIP GD alpha={1}, beta={0.1}',
        wandb_dir: str = '/workspace/saved/wandb',
        arcface_ckpt: str = 'saved/models/model_ir_se50.pth',
    ):
        self.inferencer = inferencer
        self.loader = loader
        self.device = device

        self.prompts = prompts
        self.neutral_prompts = neutral_prompts
        if self.neutral_prompts is not None:
            assert len(self.prompts) == len(self.neutral_prompts)

        self._setup_logging(wandb_project, wandb_run_name, wandb_dir)

        self.fid = FrechetInceptionDistance(device=device)
        self.ids = IDLoss(arcface_ckpt)
        self.clip_score = CLIPScore(1024, self.inferencer.clip_model, device)
        
        self.real_batches = []
        self.fake_batches = []
        
        self.fid_values = []
        self.ids_values = []
        self.clip_score_values = []

    @staticmethod
    def _setup_logging(wandb_project, wandb_run_name, wandb_dir):
        wandb.login()
        wandb.init(project=wandb_project, name=wandb_run_name, dir=wandb_dir)

    @torch.inference_mode()
    def run(self):
        for i in range(len(self.prompts)):
            text_prompt = self.prompts[i]
            if self.neutral_prompts is not None:
                neutral_prompt = self.neutral_prompts[i]
            else:
                neutral_prompt = None
            print(f'Manipulating with prompt: {text_prompt}.')
            for batch in tqdm(self.loader):
                generated_images, images = self.inferencer(
                    **batch,
                    text_prompt=text_prompt,
                    neutral_prompt=neutral_prompt
                )                
                
                self.fid.update(torch.clamp(images * 0.5 + 0.5, 0, 1),
                                is_real=True)
                self.fid.update(torch.clamp(generated_images * 0.5 + 0.5, 0, 1),
                                is_real=False)

                self.ids.update(generated_images, images)
                
                self.clip_score.update(generated_images, text_prompt)
                
                if len(self.real_batches) < 1:
                    self.real_batches.append(images.detach().cpu().squeeze())
                    self.fake_batches.append(generated_images.detach().cpu().squeeze())

            real_images = F.interpolate(torch.cat(self.real_batches, dim=0), scale_factor=0.25)
            fake_images = F.interpolate(torch.cat(self.fake_batches, dim=0), scale_factor=0.25)
            
            fake_grid = vutils.make_grid(fake_images, normalize=True, padding=2)
            
            log_dict = {
                'text_prompt': text_prompt,
                'manipulated_images': wandb.Image(fake_grid, f'{text_prompt}'),
                'FID': self.fid.compute(),
                'IDS': self.ids.compute(),
                'CLIPScore': self.clip_score.compute()
            }
            
            if i ==  0:
                real_grid = vutils.make_grid(real_images, normalize=True, padding=2)
                log_dict['source_images'] = wandb.Image(real_grid)

            wandb.log(log_dict)

            self.fid_values += [self.fid.compute()]
            self.ids_values += [self.ids.compute()]
            self.clip_score_values += [self.clip_score.compute()]
            
            self.fid.reset()
            self.ids.reset()
            self.clip_score.reset()
            
            self.real_batches.clear()
            self.fake_batches.clear()

        wandb.log({
            'FID_mean': sum(self.fid_values) / len(self.fid_values),
            'IDS_mean': sum(self.ids_values) / len(self.ids_values),
            'CLIPScore_mean': sum(self.clip_score_values) / len(self.clip_score_values)
        })
