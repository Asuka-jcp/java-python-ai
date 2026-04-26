from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_mid_slices(image_2ch: np.ndarray, out_path: Path) -> None:
    ct = image_2ch[0]
    pet = image_2ch[1]
    z = ct.shape[0] // 2
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(ct[z], cmap="gray")
    axes[0].set_title("CT mid-slice")
    axes[1].imshow(pet[z], cmap="hot")
    axes[1].set_title("PET mid-slice")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
