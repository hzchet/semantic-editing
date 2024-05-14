import os

import torch
import wandb
from tqdm import tqdm

from src.utils import inf_loop


class DeltaEditTrainer:
    def __init__(
        self,
        model,
        train_loader,
        device,
        optimizer,
        n_epochs: int,
        l2_lambda: float = None,
        cos_lambda: float = None,
        len_epoch: int = None,
        log_step: int = 50,
        save_dir: str = '/workspace/saved',
        save_every: int = 5,
        wandb_project: str = 'semantic-editing',
        wandb_run_name: str = f'DeltaEdit_baseline_training',
        wandb_dir: str = '/workspace/saved/wandb',
    ):
        self.device = device
        self.model = model.to(self.device)
        
        self.optimizer = optimizer
        self.l2_loss = torch.nn.MSELoss().to(self.device)
        self.cos_loss = torch.nn.CosineSimilarity(dim=-1).to(self.device)
        
        self.l2_lambda = l2_lambda
        self.cos_lambda = cos_lambda
        
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
        
        self.l2_losses = []
        self.cos_losses = []
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
    
    def resume_from_checkpoint(self, ckpt_path):
        state = torch.load(ckpt_path)
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])
        
        print('State loaded from checkpoint.')
        
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

            latent_s, delta_c, delta_s, latent_c = batch
            
            latent_s = latent_s.to(self.device)
            delta_c = delta_c.to(self.device)
            delta_s = delta_s.to(self.device)
            latent_c = latent_c.to(self.device)
            
            fake_delta_s = self.model(latent_s, torch.cat([latent_c, delta_c], dim=1))
            l2_loss = self.l2_loss(fake_delta_s, delta_s)
            cos_loss = 1 - torch.mean(self.cos_loss(fake_delta_s, delta_s))
            
            self.optimizer.zero_grad()
            total_loss = self.l2_lambda * l2_loss + self.cos_lambda * cos_loss
            total_loss.backward()
            self.optimizer.step()
            
            self.l2_losses += [l2_loss.detach().cpu().numpy()]
            self.cos_losses += [cos_loss.detach().cpu().numpy()]
            self.total_losses += [total_loss.detach().cpu().numpy()]
            
            if batch_idx % self.log_step == 0:
                step = epoch * self.len_epoch + batch_idx
                mean_l2_loss = sum(self.l2_losses) / len(self.l2_losses)
                mean_cos_loss = sum(self.cos_losses) / len(self.cos_losses)
                mean_total_loss = sum(self.total_losses) / len(self.total_losses)
                
                wandb.log({
                    "l2_loss": mean_l2_loss,
                    "cos_loss": mean_cos_loss,
                    "total_loss": mean_total_loss
                }, step=step)
                
                self.l2_losses.clear()
                self.cos_losses.clear()
                self.total_losses.clear()
