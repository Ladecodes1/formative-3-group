import matplotlib.pyplot as plt
import numpy as np

from em_algorithm import gaussian_pdf


def plot_fit(heights, params, bins=30):
    grid = np.linspace(heights.min() - 2, heights.max() + 2, 400)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(heights, bins=bins, density=True, alpha=0.4, color="gray", label="Heights")
    total = np.zeros_like(grid)
    for k in range(2):
        curve = params["pi"][k] * gaussian_pdf(grid, params["mu"][k], params["var"][k])
        total += curve
        ax.plot(grid, curve, label=f"Component {k + 1} (mu={params['mu'][k]:.2f})")
    ax.plot(grid, total, color="black", linestyle="--", label="Mixture")
    ax.axvline(heights.mean(), color="red", linestyle=":", label="Global mean")
    ax.set_xlabel("Height (inches)")
    ax.set_ylabel("Density")
    ax.set_title("EM fit: two-component Gaussian mixture")
    ax.legend()
    return fig


def plot_log_likelihood(history):
    iterations = [state["iteration"] for state in history]
    values = [state["log_likelihood"] for state in history]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(iterations, values, marker="o", markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log-likelihood")
    ax.set_title("Log-likelihood across EM iterations")
    return fig