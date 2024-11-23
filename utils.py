import torch
import matplotlib.pyplot as plt
import os

def unnormalize(img, mean, std):
    mean = torch.tensor(mean).reshape(1, -1, 1, 1).to(img.device)
    std = torch.tensor(std).reshape(1, -1, 1, 1).to(img.device)
    return img * std + mean

def save_images(original, reconstructed, step, output_dir, mean, std):
    os.makedirs(output_dir, exist_ok=True)

    original = unnormalize(original[:8], mean, std)
    reconstructed = unnormalize(reconstructed[:8], mean, std)

    original = original.permute(0, 2, 3, 1).cpu().detach().numpy()
    reconstructed = reconstructed.permute(0, 2, 3, 1).cpu().detach().numpy()

    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for i in range(8):
        axes[0, i].imshow(original[i].clip(0, 1))
        axes[0, i].axis("off")
        axes[1, i].imshow(reconstructed[i].clip(0, 1))
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"step_{step}.png"))
    plt.close()
