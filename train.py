import os
from tqdm.auto import trange
import torch
import hydra
from omegaconf import DictConfig
from dataloader import get_dataloader
from models.vq_autoencoder import SimpleVQAutoEncoder
from utils import save_images


@hydra.main(version_base=None, config_path="config", config_name="train_autoencoder")
def main(cfg: DictConfig):
    # Load DataLoader
    dataloader = get_dataloader(cfg.dataset, cfg.dataloader)

    # Load encoder and decoder
    encoder_cls = hydra.utils.instantiate(cfg.model.encoder)
    decoder_cls = hydra.utils.instantiate(cfg.model.decoder)

    # Initialize model
    model = SimpleVQAutoEncoder(
        encoder=encoder_cls,
        decoder=decoder_cls,
        h_dim=cfg.model.h_dim,
        num_quantizers=cfg.model.num_quantizers,
        codebook_size=cfg.model.codebook_size,
    )

    # Training loop
    train(
        model=model,
        dataloader=dataloader,
        train_iterations=cfg.train.train_iterations,
        alpha=cfg.train.alpha,
        save_every=cfg.train.save_every,
        output_dir=cfg.output_dir,
    )


def train(model, dataloader, train_iterations, alpha, save_every, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Extract normalization parameters from DataLoader
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
