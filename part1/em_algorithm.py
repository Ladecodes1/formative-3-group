import numpy as np


def gaussian_pdf(x, mean, variance):
    return np.exp(-((x - mean) ** 2) / (2 * variance)) / np.sqrt(2 * np.pi * variance)


def initialize_params(x, means=None):
    if means is None:
        means = [x.mean() - x.std(), x.mean() + x.std()]
    return {
        "mu": np.array(means, dtype=float),
        "var": np.array([x.var(), x.var()]),
        "pi": np.array([0.5, 0.5]),
    }


def weighted_densities(x, params):
    return params["pi"] * gaussian_pdf(x[:, None], params["mu"], params["var"])


def log_likelihood(x, params):
    return float(np.log(weighted_densities(x, params).sum(axis=1)).sum())


def e_step(x, params):
    weighted = weighted_densities(x, params)
    return weighted / weighted.sum(axis=1, keepdims=True)


def m_step(x, responsibilities):
    n_k = responsibilities.sum(axis=0)
    mu = (responsibilities * x[:, None]).sum(axis=0) / n_k
    var = (responsibilities * (x[:, None] - mu) ** 2).sum(axis=0) / n_k
    return {"mu": mu, "var": var, "pi": n_k / len(x)}


def record_state(iteration, x, params):
    return {"iteration": iteration, **params, "log_likelihood": log_likelihood(x, params)}


def fit_em(x, params=None, max_iter=200, tol=1e-6):
    if params is None:
        params = initialize_params(x)
    history = [record_state(0, x, params)]
    for iteration in range(1, max_iter + 1):
        responsibilities = e_step(x, params)
        params = m_step(x, responsibilities)
        history.append(record_state(iteration, x, params))
        if abs(history[-1]["log_likelihood"] - history[-2]["log_likelihood"]) < tol:
            break
    return params, history


def predict_posterior(height, params):
    return e_step(np.array([float(height)]), params)[0]