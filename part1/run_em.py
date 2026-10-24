from data_prep import build_mixture, load_galton
from em_algorithm import fit_em, predict_posterior
from reporting import build_tracking_table

heights, labels = build_mixture(load_galton())
params, history = fit_em(heights)

print(build_tracking_table(history).to_string(index=False))
print("Final iteration:", history[-1]["iteration"])
print("Final mu:", params["mu"].round(2), "pi:", params["pi"].round(3))

test_height = float(input("Enter a test height in inches: "))
posterior = predict_posterior(test_height, params)
print(f"Height {test_height}: P(group 1, lower) = {posterior[0]:.4f}, P(group 2, upper) = {posterior[1]:.4f}")