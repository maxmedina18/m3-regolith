import numpy as np
import pandas as pd

from lunar_m3.models.gmm_clustering import fit_gmm_clusters


def test_gmm_probabilities_sum_to_one() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": np.arange(200),
            "y": np.zeros(200, dtype=int),
            "bd_1um": rng.normal(0.03, 0.01, size=200),
            "bc_1um": rng.normal(1.00, 0.05, size=200),
            "bd_2um": rng.normal(0.02, 0.01, size=200),
            "bc_2um": rng.normal(2.10, 0.08, size=200),
            "slope_global": rng.normal(0.02, 0.01, size=200),
            "brightness_ref": rng.normal(0.10, 0.02, size=200),
        }
    )

    cols = ["bd_1um", "bc_1um", "bd_2um", "bc_2um", "slope_global", "brightness_ref"]
    res = fit_gmm_clusters(df, feature_columns=cols, n_clusters=4, random_state=0)
    sums = np.sum(res.probabilities, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6)

