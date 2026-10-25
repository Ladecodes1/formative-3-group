# Formative 3: Probability, Bayes and Gradient Descent

Group project covering four parts: the EM algorithm on a mixture of Gaussians, Bayes' theorem on IMDb reviews, gradient descent by hand, and gradient descent in Python.

## Team

| Member          | Main responsibility                                    |
| --------------- | ------------------------------------------------------ |
| Sylivie       | Part 1: Probability distributions and the EM algorithm |
| Naomi           | Part 2: Bayesian probability on IMDb reviews           |
| Byiringiro Saad | Parts 3 and 4: Gradient descent by hand and in code    |

Every member completes at least one of the three manual gradient descent updates in Part 3, reviews another member's work, and presents.

## Repository

| File                            | Contents                                             | Status |
| ------------------------------- | ---------------------------------------------------- | ------ |
| `part1/part1_em.ipynb`          | EM algorithm from scratch (helper files in `part1/`) | Done   |
| `part2_bayes_imdb.ipynb`        | Bayes' theorem on IMDb keywords                      | Done   |
| `part3_manual_calculations.pdf` | Handwritten gradient descent updates                 | Done   |
| `part4_gradient_descent.ipynb`  | Gradient descent in Python                           | Done   |
| `images/`                       | Plots used in this README                            |        |

## How to run

**In Google Colab:** open a notebook on GitHub and click its _Open in Colab_ button.

**Locally:**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib jupyter pandas
jupyter notebook
```

Part 2 needs `IMDB Dataset.csv` (50,000 labelled reviews) in the repository folder; it is not committed because of its size. Part 4 needs no data file.

## Part 1: EM algorithm

### Overview

The heights of children and fathers from the Galton families dataset are treated as an unlabeled mixture of two Gaussian distributions. The Expectation-Maximization (EM) algorithm is implemented from scratch with NumPy to estimate the means, variances and mixing weights of the two groups.

### Data

- Source: Galton's parents' and children's heights (`part1/GaltonFamilies.csv`), in inches.
- The file has one row per child, so each father's height repeats for every child in his family. Fathers are de-duplicated by family (205 fathers) and combined with all 934 children, giving 1139 unlabeled heights.
- The true labels are kept aside for checking only. EM never sees them.

### Files (in `part1/`)

| File              | Purpose                                                                                             |
| ----------------- | --------------------------------------------------------------------------------------------------- |
| `data_prep.py`    | Loads the CSV and builds the unlabeled heights array and the label array                            |
| `em_algorithm.py` | EM implementation: pdf, E-step, M-step, log-likelihood, fitting loop, posterior for a single height |
| `reporting.py`    | Builds the optimization tracking table                                                              |
| `mean_split.py`   | Measures the global-mean split against EM                                                           |
| `plots.py`        | Fit plot and log-likelihood plot                                                                    |
| `run_em.py`       | Terminal driver: prints the table and classifies a test height                                      |
| `part1_em.ipynb`  | Notebook with all results, tables and plots                                                         |

### How to run

Dependencies: numpy, pandas, matplotlib, ipykernel.

```
py -m pip install numpy pandas matplotlib ipykernel
cd part1
py run_em.py
```

For the notebook, open `part1/part1_em.ipynb` in VS Code and use Run All.

Part 1 imports its helper files from the same folder, so it must run with the `part1/` folder next to it. In Colab, clone the repository first:

```
!git clone https://github.com/Ladecodes1/formative-3-group.git
%cd formative-3-group/part1
```

### Method

1. **Initialization:** means placed one standard deviation either side of the overall mean, equal variances, mixing weights 0.5 each.
2. **E-step:** responsibilities (soft assignments) from the current means, variances and weights.
3. **M-step:** weighted updates of the means, variances and mixing weights using the responsibilities.
4. **Convergence:** stop when the log-likelihood changes by less than 1e-6 or after 200 iterations.

### Optimization tracking table

| Iteration          | mu1     | mu2     | var1    | var2    | pi1    | pi2    | Log-Likelihood |
| ------------------ | ------- | ------- | ------- | ------- | ------ | ------ | -------------- |
| 0 (Initialization) | 63.6414 | 70.7756 | 12.7241 | 12.7241 | 0.5000 | 0.5000 | -3194.5755     |
| 1                  | 64.9339 | 69.4553 | 7.9124  | 7.3183  | 0.4969 | 0.5031 | -3053.3088     |
| 2                  | 64.8745 | 69.5053 | 7.7363  | 6.9965  | 0.4960 | 0.5040 | -3052.1871     |

Component 1 is the lower-height group and component 2 the upper-height group. After 200 iterations the means are 64.19 and 69.53 and the weights are 0.435 and 0.565.

### Test height classification

`predict_posterior(height, params)` returns the posterior probability of each group. Example: a height of 67 in gives P(group 1) = 0.3995 and P(group 2) = 0.6005. Heights must be in inches.

### Why not split at the global mean?

- The global mean is 67.21 in. Splitting there placed 403 of 934 children in the upper pile and 44 of 205 fathers in the lower pile, with 60.8% agreement with the child/father labels.
- A hard split forces everyone into one group and ignores the groups' different sizes and spreads. EM gives a probability instead: 309 of the 1139 heights have a posterior between 0.2 and 0.8 and are genuinely uncertain.

### Note on the groups

EM finds the structure in the heights, not the child/father labels. The children are themselves a mix of daughters and sons, so the two components correspond mostly to the lower and upper height groups.

## Part 2: Bayesian probability

We compute **P(Positive | keyword)** for 4 positive and 4 negative keywords, using only Python's standard library. Each word counts once per review. The prior is P(Positive) = 0.5 (25,000 of 50,000 reviews).

| Keyword   | Likelihood P(kw \| Pos) | Marginal P(kw) | Posterior P(Pos \| kw) |
| --------- | ----------------------- | -------------- | ---------------------- |
| superb    | 0.0410                  | 0.0240         | 0.8557                 |
| wonderful | 0.0902                  | 0.0555         | 0.8120                 |
| excellent | 0.1146                  | 0.0710         | 0.8073                 |
| brilliant | 0.0634                  | 0.0417         | 0.7606                 |
| boring    | 0.0236                  | 0.0610         | 0.1938                 |
| awful     | 0.0112                  | 0.0575         | 0.0977                 |
| worst     | 0.0164                  | 0.0886         | 0.0924                 |
| waste     | 0.0070                  | 0.0507         | 0.0687                 |

Every positive keyword lifts the chance of a positive review well above 0.5, and every negative keyword pushes it well below.

## Part 3: Gradient descent by hand

Model: y = m₁x₁ + m₂x₂ + b, written as ŷ = Xm + b, with X = [[1, 3], [4, 10]], y = [5, 6], starting m = [−1, 2] and b = [1, 1].

From the MSE cost J = (1/n)·Σ(ŷ − y)², the chain rule gives:

- ∂J/∂m = (2/n)·Xᵀ(ŷ − y)
- ∂J/∂b = (2/n)·Σ(ŷ − y)

Agreed conventions:

- Learning rate α = 0.005. The brief gives none; above about 0.0157 the cost grows instead of shrinking.
- b is one shared bias written once per data point, so both entries always stay equal.
- New m and b are rounded to 4 decimal places after each update.

| After    | m₁      | m₂     | b      | J       | Done by         |
| -------- | ------- | ------ | ------ | ------- | --------------- |
| Start    | −1.0000 | 2.0000 | 1.0000 | 61.0000 |                 |
| Update 1 | −1.2250 | 1.4350 | 0.9400 | 10.1163 | Byiringiro Saad |
| Update 2 | −1.3079 | 1.2302 | 0.9230 | 3.4221  | Person 1        |
| Update 3 | −1.3393 | 1.1559 | 0.9215 | 2.5323  | Naomi           |

## Part 4: Gradient descent in code

The notebook repeats the Part 3 updates in NumPy with matrix multiplication and prints every step. `derivative(equation, params)` uses SciPy's `approx_fprime` to differentiate the cost equation, and it matches the derived formulas at every update. The printed numbers match the handwritten ones exactly.

![m and b over iterations](images/part4_parameters.png)

m₂ moves the most because its feature x₂ has the largest values (3 and 10), giving it the largest gradient.

![Cost over iterations](images/part4_cost.png)

**The updates reduce the error every time:** J falls from 61 to 2.5323, 96% lower after three updates. The steps shrink as the cost flattens near its minimum. The final predictions are [3.0499, 7.1233] against targets [5, 6].

## Presentation plan

| Order | Topic                                                                        | Presenter       |
| ----- | ---------------------------------------------------------------------------- | --------------- |
| 1     | Part 1: EM algorithm, tracking table, live test height, global-mean question | Sylivie         |
| 2     | Part 2: keyword choices and Bayes tables                                     | Naomi           |
| 3     | Parts 3 and 4: derivation, manual updates, code and plots                    | Byiringiro Saad |

The full presentation is 15 minutes, followed by questions for any member.
