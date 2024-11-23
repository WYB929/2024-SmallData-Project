from torch.utils.data import DataLoader
from dataset import get_dataset
from omegaconf import DictConfig

def get_dataloader(dataset_cfg: DictConfig, dataloader_cfg: DictConfig):
    dataset = get_dataset(dataset_cfg)
    return DataLoader(
        dataset,
        batch_size=dataloader_cfg.batch_size,
        shuffle=dataloader_cfg.shuffle,
        num_workers=dataloader_cfg.num_workers,
    )
