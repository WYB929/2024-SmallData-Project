import os
from tqdm.auto import trange
import torch
import hydra
from omegaconf import DictConfig
from dataloader import get_dataloader
from models.resnet_classifier import ResNetClassifier


@hydra.main(version_base=None, config_path="config", config_name="train_classifier")
def main(cfg: DictConfig):
    # Load DataLoaders
    train_loader = get_dataloader(cfg.dataset.train, cfg.dataloader)
    val_loader = get_dataloader(cfg.dataset.val, cfg.dataloader)

    # Load encoder
    encoder_cls = hydra.utils.instantiate(cfg.model.encoder)

    # Initialize ResNetClassifier
    model = ResNetClassifier(
        encoder=encoder_cls,
        num_classes=cfg.model.num_classes,
        h_dim=cfg.model.h_dim,
    )

    # Training loop
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_iterations=cfg.train.train_iterations,
        save_every=cfg.train.save_every,
        eval_every=cfg.train.eval_every,
        output_dir=cfg.output_dir,
        learning_rate=cfg.train.learning_rate,
    )


def train(model, train_loader, val_loader, train_iterations, save_every, eval_every, output_dir, learning_rate):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in (pbar := trange(train_iterations)):
        model.train()
        x, labels = next(iter(train_loader))
        x, labels = x.to(device), labels.to(device)

        opt.zero_grad()
        loss, logits = model.compute_loss(x, labels)
        loss.backward()
        opt.step()

        pbar.set_description(
            f"Step: {step} | Loss: {loss.item():.4f}"
        )

        # Evaluate the model periodically
        if step % eval_every == 0 or step == train_iterations - 1:
            val_loss, val_accuracy = model.eval_model(val_loader, device)
            print(f"Step: {step} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4%}")

        # Save checkpoint periodically
        if step % save_every == 0 or step == train_iterations - 1:
            torch.save(model.state_dict(), os.path.join(output_dir, f"model_step_{step}.pth"))


if __name__ == "__main__":
    main()
