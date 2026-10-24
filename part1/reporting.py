import pandas as pd


def iteration_label(iteration):
    return "0 (Initialization)" if iteration == 0 else str(iteration)


def build_tracking_table(history, iterations=(0, 1, 2)):
    rows = [
        {
            "Iteration": iteration_label(state["iteration"]),
            "mu1": state["mu"][0],
            "mu2": state["mu"][1],
            "var1": state["var"][0],
            "var2": state["var"][1],
            "pi1": state["pi"][0],
            "pi2": state["pi"][1],
            "Log-Likelihood": state["log_likelihood"],
        }
        for state in history
        if state["iteration"] in iterations
    ]
    return pd.DataFrame(rows).round(4)