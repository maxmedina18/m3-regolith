from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_spectrum_comparison(
    wavelengths_um: np.ndarray,
    raw: np.ndarray,
    processed: np.ndarray,
    *,
    title: str,
    output_path: Optional[str | Path] = None,
) -> None:
    """Plot raw vs processed spectrum."""

    w = np.asarray(wavelengths_um, dtype=float)
    r0 = np.asarray(raw, dtype=float)
    r1 = np.asarray(processed, dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(w, r0, linewidth=1.2, label="raw")
    ax.plot(w, r1, linewidth=1.6, label="processed")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Reflectance (a.u.)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()
