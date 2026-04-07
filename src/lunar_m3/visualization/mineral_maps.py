from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_label_map(
    label_grid: np.ndarray,
    *,
    title: str,
    output_path: Optional[str | Path] = None,
) -> None:
    """Plot a classification label map."""

    labels = np.asarray(label_grid)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(labels, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="class")
    fig.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()
