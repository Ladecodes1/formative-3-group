from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).parent / "GaltonFamilies.csv"


def load_galton(path=DATA_PATH):
    return pd.read_csv(path)


def build_mixture(df, parent="father"):
    children = df["childHeight"].to_numpy(dtype=float)
    parents = df.drop_duplicates(subset="family")[parent].to_numpy(dtype=float)
    heights = np.concatenate([children, parents])
    labels = np.concatenate(
        [np.zeros(len(children), dtype=int), np.ones(len(parents), dtype=int)]
    )
    return heights, labels


if __name__ == "__main__":
    df = load_galton()
    heights, labels = build_mixture(df)
    print("Total heights:", len(heights))
    print("Children:", int((labels == 0).sum()), "mean", round(heights[labels == 0].mean(), 2))
    print("Fathers:", int((labels == 1).sum()), "mean", round(heights[labels == 1].mean(), 2))
    print("Global mean:", round(heights.mean(), 2))