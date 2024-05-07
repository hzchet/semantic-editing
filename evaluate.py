import warnings

import torch
import click
from addict import Dict
from torch.utils.data import DataLoader

from src.utils import load_yaml, seed_everything, build_object
from src.trainers import Evaluator
warnings.filterwarnings("ignore", category=UserWarning)


@click.command()
@click.argument('cfg_path', type=click.Path(), default=None)
def main(cfg_path):
    cfg = Dict(load_yaml(cfg_path))
    seed_everything(cfg.seed)
    
    device = torch.device(cfg.device)
    
    inferencer = build_object(
        cfg.inferencer.type,
        cfg.inferencer.module
    )(device=device, **cfg.inferencer.params)
    
    dataset = build_object(
        cfg.data.type,
        cfg.data.module
    )(**cfg.data.params)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )
    
    evaluator = Evaluator(
        inferencer,
        loader,
        device,
        **cfg.evaluator.params
    )
    
    evaluator.run()


if __name__ == '__main__':
    main()
