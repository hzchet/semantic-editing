import random

import torch
import numpy as np
from torch.utils.data import Dataset


class TrainDeltaFFHQ(Dataset):
    def __init__(
        self,
        path_to_w_latents: str,
        path_to_s_latents: str = None,
        path_to_c_latents: str = None,
        path_to_noise_w_latents: str = None,
        path_to_noise_s_latents: str = None,
        path_to_noise_c_latents: str = None,
        limit_real: int = 58_000,
        limit_noise: int = 200_000
    ):
        super().__init__()

        assert 0 < limit_real <= 58000
        assert 0 < limit_noise <= 200_000
        self.w_latents = torch.from_numpy(np.load(path_to_w_latents)[:58000][:limit_real])

        if path_to_s_latents is not None:
            self.s_latents = torch.from_numpy(np.load(path_to_s_latents)[:58000][:limit_real])
        else:
            self.s_latents = None
            
        if path_to_c_latents is not None:
            self.c_latents = torch.from_numpy(np.load(path_to_c_latents)[:58000][:limit_real])
        else:
            self.c_latents = None

        if path_to_noise_w_latents is not None:
            noise_w_latents = torch.from_numpy(np.load(path_to_noise_w_latents)[:limit_noise])
            self.w_latents = torch.cat([self.w_latents, noise_w_latents], dim=0)
        if path_to_noise_s_latents is not None:
            noise_s_latents = torch.from_numpy(np.load(path_to_noise_s_latents)[:limit_noise])
            self.s_latents = torch.cat([self.s_latents, noise_s_latents], dim=0)
        if path_to_noise_c_latents is not None:
            noise_c_latents = torch.from_numpy(np.load(path_to_noise_c_latents)[:limit_noise])
            self.c_latents = torch.cat([self.c_latents, noise_c_latents], dim=0)

    def __len__(self):
        return len(self.w_latents)
    
    def __getitem__(self, idx: int):
        latent_w1 = self.w_latents[idx]
        latent_s1 = self.s_latents[idx]
        latent_c1 = self.c_latents[idx]
        latent_c1 = latent_c1 / latent_c1.norm(dim=-1, keepdim=True).float()
        
        rand_idx = random.randint(0, self.__len__() - 1)
        latent_w2 = self.w_latents[rand_idx]
        latent_s2 = self.s_latents[rand_idx]
        latent_c2 = self.c_latents[rand_idx]
        latent_c2 = latent_c2 / latent_c2.norm(dim=-1, keepdim=True).float()

        delta_w = latent_w2 - latent_w1        
        delta_s = latent_s2 - latent_s1

        delta_c = latent_c2 - latent_c1
        delta_c = delta_c / delta_c.norm(dim=-1, keepdim=True).float().clamp(min=1e-5)
        
        return latent_s1, delta_c, delta_s, latent_c1, latent_w1


class EvalFFHQ(Dataset):
    def __init__(
        self,
        path_to_w_latents: str,
        path_to_s_latents: str = None,
        path_to_c_latents: str = None,
        limit: int = 12000
    ):
        super().__init__()

        assert 0 < limit <= 12000
        
        self.w_latents = torch.from_numpy(np.load(path_to_w_latents)[58000:][:limit])

        if path_to_s_latents is not None:
            self.s_latents = torch.from_numpy(np.load(path_to_s_latents)[58000:][:limit])
        else:
            self.s_latents = None
            
        if path_to_c_latents is not None:
            self.c_latents = torch.from_numpy(np.load(path_to_c_latents)[58000:][:limit])
        else:
            self.c_latents = None

    def __len__(self):
        return len(self.w_latents)

    def __getitem__(self, idx: int):
        item = {}
        
        item['w_latent'] = self.w_latents[idx]
        
        if self.s_latents is not None:
            item['s_latent'] = self.s_latents[idx]
            
        if self.c_latents is not None:
            item['c_latent'] = self.c_latents[idx]
            item['c_latent'] = item['c_latent'] / item['c_latent'].norm(dim=-1, keepdim=True).float()
            
        return item
