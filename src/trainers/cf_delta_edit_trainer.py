import os

import torch
import wandb
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms

from src.utils import inf_loop
from src.models.delta_edit.utils import decoder_validate
from src.losses import NCELoss, IDLoss
from src.models.stylegan2 import Generator



class CFDeltaEditTrainer:
    def __init__(
        self,
        model,
        train_loader,
        device,
        optimizer,
        stylegan2_ckpt: str,
        n_epochs: int,
        l2_lambda: float = None,
        cos_lambda: float = None,
        id_lambda: float = None,
        nce_lambda: float = None,
        num_aug: int = 8,
        len_epoch: int = None,
        log_step: int = 50,
        src_text: str = 'face',
        save_dir: str = '/workspace/saved',
        save_every: int = 5,
        wandb_project: str = 'semantic-editing',
        wandb_run_name: str = f'CF-DeltaEdit_training',
        wandb_dir: str = '/workspace/saved/wandb',
    ):
        self.device = device
        self.model = model.to(self.device)
        
        self.optimizer = optimizer
        self.l2_loss = torch.nn.MSELoss().to(self.device)
        self.cos_loss = torch.nn.CosineSimilarity(dim=-1).to(self.device)
        self.nce_loss = NCELoss(device, src_text)
        self.id_loss = IDLoss(device)
        
        self.augment = transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5)
        self.num_aug = num_aug
        
        self.l2_lambda = l2_lambda
        self.cos_lambda = cos_lambda
        self.id_lambda = id_lambda
        self.nce_lambda = nce_lambda
        
        self.n_epochs = n_epochs
        self.train_loader = train_loader
        if len_epoch is None:
            self.len_epoch = len(train_loader)
        else:
            self.train_loader = inf_loop(self.train_loader)
            self.len_epoch = len_epoch
        
        self.save_path = os.path.join(save_dir, wandb_run_name)
        os.makedirs(os.path.join(save_dir, wandb_run_name), exist_ok=True)
        self.save_every = save_every
        self.log_step = log_step
        
        self._setup_logging(wandb_project, wandb_run_name, wandb_dir)
        
        self.generator = Generator(
            size=1024,
            style_dim=512,
            n_mlp=8
        )
        ckpt = torch.load(stylegan2_ckpt)
        self.generator.load_state_dict(ckpt['g_ema'], strict=False)
        self.generator.eval()
        self.generator = self.generator.to(device)
        for param in self.generator.parameters():
            param.requires_grad = False
        
        self.l2_losses = []
        self.cos_losses = []
        self.nce_losses = []
        self.id_losses = []
        self.total_losses = []
        
    @staticmethod
    def _setup_logging(wandb_project, wandb_run_name, wandb_dir):
        wandb.login()
        wandb.init(project=wandb_project, name=wandb_run_name, dir=wandb_dir)
    
    def train(self, start_epoch: int = 0):
        try:
            for epoch in range(start_epoch, self.n_epochs):
                self.train_epoch(epoch)
                if epoch % self.save_every == 0:
                    self.save_checkpoint(epoch)
        except KeyboardInterrupt as e:
            print('Saving model on keyboard interrupt...')
            self.save_checkpoint(epoch)
            raise e
        
        if self.save_every != 1:
            self.save_checkpoint(self.n_epochs)
        
    def save_checkpoint(self, epoch):
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        filename = os.path.join(self.save_path, f'epoch_{epoch}.ckpt')
        print('Saving checkpoint...')
        torch.save(state, filename)
    
    def resume_from_checkpoint(self, ckpt_path, finetune: bool = False):
        state = torch.load(ckpt_path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        
        print('State loaded from checkpoint.')
        if finetune:
            self.train()
        else:
            self.train(state['epoch'])

    def train_epoch(self, epoch):
        self.model.train()
        
        for batch_idx, batch in enumerate(
            tqdm(
                self.train_loader, desc=f'Training epoch {epoch + 1}/{self.n_epochs}',
                total=self.len_epoch
            )
        ):
            if batch_idx == self.len_epoch:
                break

            latent_s, delta_c, delta_s, latent_c, latent_w = batch
            
            latent_s = latent_s.to(self.device)
            delta_c = delta_c.to(self.device)
            delta_s = delta_s.to(self.device)
            latent_c = latent_c.to(self.device)
            latent_w = latent_w.to(self.device)
            
            fake_delta_s = self.model(latent_s, torch.cat([latent_c, delta_c], dim=1))
            l2_loss = self.l2_loss(fake_delta_s, delta_s)
            # cos_loss = 1 - torch.mean(self.cos_loss(fake_delta_s, delta_s))
            
            x = decoder_validate(self.generator, latent_s, latent_w)
            x_hat = decoder_validate(self.generator, fake_delta_s + latent_s, latent_w)
            x_aug = torch.cat([self.augment(x_hat) for _ in range(self.num_aug)], dim=0)

            nce_loss = self.nce_loss(
                x_aug, 
                latent_c.repeat(self.num_aug, 1), 
                delta_c.repeat(self.num_aug, 1)
            )
            id_loss = self.id_loss(x_hat, x)
            
            self.optimizer.zero_grad()
            total_loss = self.l2_lambda * l2_loss + self.id_lambda * id_loss +\
                         self.nce_lambda * nce_loss
            total_loss.backward()
            self.optimizer.step()
            
            self.l2_losses += [l2_loss.detach().cpu().numpy()]
            # self.cos_losses += [cos_loss.detach().cpu().numpy()]
            self.id_losses += [id_loss.detach().cpu().numpy()]
            self.total_losses += [total_loss.detach().cpu().numpy()]
            self.nce_losses += [nce_loss.detach().cpu().numpy()]
            
            if batch_idx % self.log_step == 0:
                step = epoch * self.len_epoch + batch_idx
                mean_l2_loss = sum(self.l2_losses) / len(self.l2_losses)
                # mean_cos_loss = sum(self.cos_losses) / len(self.cos_losses)
                mean_id_loss = sum(self.id_losses) / len(self.id_losses)
                mean_nce_loss = sum(self.nce_losses) / len(self.nce_losses)
                mean_total_loss = sum(self.total_losses) / len(self.total_losses)
                
                wandb.log({
                    "l2_loss": mean_l2_loss,
                    # "cos_loss": mean_cos_loss,
                    "id_loss": mean_id_loss,
                    "total_loss": mean_total_loss,
                    "nce_loss": mean_nce_loss
                }, step=step)
                
                self.l2_losses.clear()
                # self.cos_losses.clear()
                self.id_losses.clear()
                self.nce_losses.clear()
                self.total_losses.clear()
