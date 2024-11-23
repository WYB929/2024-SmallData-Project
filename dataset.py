from torchvision import datasets, transforms
import hydra
from omegaconf import DictConfig

def get_dataset(cfg: DictConfig):
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std),
    ])
    return datasets.ImageFolder(root=cfg.root, transform=transform)
