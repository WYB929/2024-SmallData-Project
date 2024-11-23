import os
from tqdm.auto import trange
import torch
import hydra
from omegaconf import DictConfig
from dataloader import get_dataloader
from models.vq_autoencoder import SimpleVQAutoEncoder
from utils import save_images
from hydra.core.global_hydra import GlobalHydra
@hydra.main(version_base=None, config_path="config", config_name="train_autoencoder")
def main(cfg: DictConfig):
    # Load DataLoader
    print(cfg)
    dataloader = get_dataloader(cfg.dataset, cfg.dataloader)

    # Initialize model
    model_cfg = cfg.model
    model = SimpleVQAutoEncoder(
        in_dim=model_cfg.in_dim,
        h_dim=model_cfg.h_dim,
        res_h_dim=model_cfg.res_h_dim,
        n_res_layers=model_cfg.n_res_layers,
        num_quantizers=model_cfg.num_quantizers,
        codebook_size=model_cfg.codebook_size,
    )

    # Training loop
    train_cfg = cfg.train
    train(
        model,
        dataloader,
        train_iterations=train_cfg.train_iterations,
        alpha=train_cfg.alpha,
        save_every=train_cfg.save_every,
        output_dir=cfg.output_dir,
    )

def train(model, dataloader, train_iterations, alpha, save_every, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    mean = dataloader.dataset.transform.transforms[-1].mean
    std = dataloader.dataset.transform.transforms[-1].std

    for step in (pbar := trange(train_iterations)):
        model.train()
        x, _ = next(iter(dataloader))
        x = x.to(device)

        opt.zero_grad()
        total_loss, rec_loss, vq_loss, indices = model.compute_loss(x, alpha=alpha)
        total_loss.backward()
        opt.step()

        pbar.set_description(
            f"rec loss: {rec_loss.item():.3f} | "
            f"vq loss: {vq_loss.item():.3f} | "
            f"active %: {indices.unique().numel() / model.vq.codebook_size * 100:.3f}"
        )

        if step % save_every == 0 or step == train_iterations - 1:
            save_images(x, model(x)[0], step, output_dir, mean, std)

if __name__ == "__main__":
    main()
