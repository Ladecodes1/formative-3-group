from data_prep import build_mixture, load_galton
from em_algorithm import e_step, fit_em


def mean_split_summary(heights, labels):
    threshold = heights.mean()
    predicted = (heights > threshold).astype(int)
    lower_pile = heights[predicted == 0]
    upper_pile = heights[predicted == 1]
    return {
        "threshold": threshold,
        "pile_means": (lower_pile.mean(), upper_pile.mean()),
        "pile_sizes": (len(lower_pile), len(upper_pile)),
        "accuracy": (predicted == labels).mean(),
        "children_in_upper_pile": int(((labels == 0) & (predicted == 1)).sum()),
        "fathers_in_lower_pile": int(((labels == 1) & (predicted == 0)).sum()),
    }


def count_uncertain(heights, params, low=0.2, high=0.8):
    upper_probability = e_step(heights, params)[:, 1]
    return int(((upper_probability > low) & (upper_probability < high)).sum())


if __name__ == "__main__":
    heights, labels = build_mixture(load_galton())
    params, history = fit_em(heights)
    summary = mean_split_summary(heights, labels)
    print("Global mean threshold:", round(summary["threshold"], 2))
    print("Pile sizes (lower, upper):", summary["pile_sizes"])
    print("Pile means (lower, upper):", tuple(round(float(m), 2) for m in summary["pile_means"]))
    print("True group means (children, fathers):", round(heights[labels == 0].mean(), 2), round(heights[labels == 1].mean(), 2))
    print("Accuracy vs child/father labels:", round(summary["accuracy"], 3))
    print("Children placed in upper pile:", summary["children_in_upper_pile"], "of", int((labels == 0).sum()))
    print("Fathers placed in lower pile:", summary["fathers_in_lower_pile"], "of", int((labels == 1).sum()))
    print("EM component means:", params["mu"].round(2))
    print("Heights EM is unsure about (P between 0.2 and 0.8):", count_uncertain(heights, params))