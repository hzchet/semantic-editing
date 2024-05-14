import warnings

import torch
import click
from addict import Dict
from torch.utils.data import DataLoader

from src.utils import load_yaml, seed_everything, build_object
warnings.filterwarnings("ignore", category=UserWarning)


@click.command()
@click.argument('cfg_path', type=click.Path(), default=None)
def main(cfg_path):
    cfg = Dict(load_yaml(cfg_path))
    seed_everything(cfg.seed)
    
    device = torch.device(cfg.device)
    
    model = build_object(
        cfg.model.type,
        cfg.model.module
    )(**cfg.model.params)
    
    dataset = build_object(
        cfg.data.type,
        cfg.data.module
    )(**cfg.data.params)
    
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )
    
    optimizer = build_object(
        cfg.optimizer.type,
        cfg.optimizer.module
    )(model.parameters(), **cfg.optimizer.params)
    
    trainer = build_object(
        cfg.trainer.type,
        cfg.trainer.module
    )(
        model,
        train_loader,
        device,
        optimizer,
        **cfg.trainer.params
    )
    
    if cfg.resume:
        trainer.resume_from_checkpoint(cfg.resume_path, cfg.finetune)
    else:
        trainer.train()


if __name__ == '__main__':
    main()
