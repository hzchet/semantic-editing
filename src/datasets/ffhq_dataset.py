import torch
import numpy as np
from torch.utils.data import Dataset


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
