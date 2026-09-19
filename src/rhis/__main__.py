from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame

from rhis.core import Rhis


def generate_example_data() -> DataFrame:
    rng = np.random.default_rng(42)

    df = pd.DataFrame({
        "series_A": np.clip(rng.normal(50, 15, 100), 0, 100),
        "series_B": np.clip(rng.normal(50, 15, 100), 0, 100),
        "series_C": np.clip(rng.normal(50, 15, 100), 0, 100),
        "series_D": np.clip(rng.normal(50, 15, 100), 0, 100),
    })

    # Forcing a trend on series_A and series_B
    df["series_A"] = np.sort(df["series_A"].to_numpy())
    df["series_B"] = np.sort(df["series_B"].to_numpy())

    return df


def main() -> None:
    df = generate_example_data()
    rhis = Rhis(df)
    rhis.evol()
    rhis.add_rhis_compliant_to_df()
    rhis.plot()


if __name__ == "__main__":
    main()
