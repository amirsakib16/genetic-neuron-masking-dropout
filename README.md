# GA-AD: Importance-Guided Genetic Neuron Masking as an Alternative to Dropout

A controlled empirical study of **GA-AD** (Genetic-Algorithm Adaptive Dropout): instead of dropping hidden units uniformly at random, a genetic algorithm chooses *which* units to drop, at a fixed rate, using gradient-based importance scores. It is compared against no dropout, standard dropout, random structured masks and a deterministic importance-guided control on three benchmarks.

> **Headline result (negative):** GA-AD did **not** outperform standard dropout on any benchmark. Differences are within sampling noise. See [Results](#results) and [Limitations](#limitations).

---

## Table of Contents

- [Overview](#overview)
- [Method](#method)
- [Experimental Setup](#experimental-setup)
- [Results](#results)
- [Limitations](#limitations)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [License](#license)

---

## Overview

| Notebook | Dataset | Seeds | GA-AD variant |
|---|---|---|---|
| `breast-cancer-wisconsin-diagnostic.ipynb` | Breast Cancer Wisconsin (Diagnostic) | 5 | v1 |
| `mnist-digit-recognizer-ga-ad.ipynb` | MNIST (Kaggle `train.csv`) | 1 | v2 |
| `fashion-mnist-ga-ad.ipynb` | Fashion-MNIST | 1 | v2 |

All models use the same MLP: $d \rightarrow 512 \rightarrow 256 \rightarrow 128 \rightarrow K$ with ReLU and a softmax output.

---

## Method

### 1. Masked forward pass

Hidden layer $\ell$ has $N_\ell$ units. A binary mask $m^{(\ell)} \in \{0,1\}^{N_\ell}$ has exactly $D_\ell$ inactive units:

$$
D_\ell = \min\big(\operatorname{round}(p\,N_\ell),\; N_\ell - 1\big), \qquad p = 0.3
$$

During training, inverted-dropout scaling is applied:

$$
\tilde{h}^{(\ell)} = \frac{m^{(\ell)} \odot h^{(\ell)}}{\bar{m}^{(\ell)}}, \qquad
\bar{m}^{(\ell)} = \frac{1}{N_\ell}\sum_{i=1}^{N_\ell} m_i^{(\ell)}
$$

At inference the full network is used (no masking).

### 2. Neuron importance $Q$

**Variant v2 (MNIST, Fashion-MNIST).** For a minibatch $B$:

$$
I_i = \mathbb{E}_{B}\left[\,\left| h_i \,\frac{\partial \mathcal{L}}{\partial h_i} \right|\,\right]
$$

An exponential moving average with $\rho = 0.9$ gives $\bar{I}_i$, then robust percentile scaling:

$$
Q_i = \operatorname{clip}\!\left(\frac{\bar{I}_i - P_{5}}{P_{95} - P_{5}},\, 0,\, 1\right)
$$

**Variant v1 (Breast Cancer).** The EMA-smoothed activation gradient yields $s_i = \text{minmax}\left(\left|\Delta \text{EMA}_i\right|\right)$. A polynomial descriptor uses $a_i = 10\,s_i$:

$$
C_i = \sum_{k=1}^{2} w_k \left| (a_i)_k \right|, \qquad w_k = \tfrac{1}{2}, \qquad (a)_k = a(a-1)\cdots(a-k+1)
$$

$$
Q_i = 0.6\, s_i + 0.4\, \hat{C}_i
$$

where $\hat{C}_i$ is $C_i$ min–max normalised. Since $\hat{C}_i$ is a deterministic function of $s_i$, it adds no independent information; v2 removes it.

In both variants, **high $Q$ means keep, low $Q$ means eligible to drop.**

### 3. Genetic mask search

Fitness is the mean importance retained by the active units:

$$
F(m) = \frac{\sum_{i} Q_i\, m_i}{\max\!\left(\sum_{i} m_i,\; 1\right)}
$$

| Component | Setting |
|---|---|
| Population $P$ | 16 (v1), 20 (v2) |
| Generations $G$ | 6 |
| Elites $n_e$ | $\max(2,\lfloor 0.25P \rfloor)$ |
| Crossover | uniform, two random elites |
| Mutation | per-gene flip, rate 0.04 |
| Repair | drop lowest-$Q$ actives / add highest-$Q$ inactives to restore cardinality |
| Output | pool of up to $n_e$ distinct elite masks |

Validation data and labels **never** enter the fitness.

### 4. Training schedule

- **v1:** 4 mask refreshes per epoch (recompute $Q$, rerun the GA, sample one elite per layer).
- **v2:** elite pool rebuilt once per epoch; a different elite is sampled **per minibatch**.

### 5. Why the GA adds little

**Proposition.** For fixed cardinality $A$ and distinct $Q_i$, $F$ is maximised uniquely by activating the $A$ units with the largest $Q$.

*Proof.* With $\sum_i m_i = A$, $F(m) = \frac{1}{A}\sum_i Q_i m_i$ is a sum of $A$ distinct $Q$ values, maximal exactly when they are the $A$ largest. $\blacksquare$

The optimum is found by a sort in $O(N \log N)$, so GA-AD behaves like Q-guided masking plus stochastic perturbation. (The overlap between GA elites and the exact top-$k$ set was **not measured**.)

---

## Experimental Setup

| Dataset | Train | Val. | GA hold-out | Test |
|---|---:|---:|---:|---:|
| Breast Cancer | 341 | 114 | – | 114 |
| MNIST | 26,458 | 6,302 | 2,940 | 6,300 |
| Fashion-MNIST | 42,000 | 10,000 | 8,000 | 10,000 |

- **Optimiser:** Adam, learning rate $10^{-3}$, categorical cross-entropy
- **Early stopping:** validation loss, patience 6, best weights restored
- **Batch size / epoch cap:** 32/40 (Breast Cancer), 128/40 baselines and 128/25 custom (MNIST), 128/35 (Fashion-MNIST)
- **Seeds (Breast Cancer):** 42, 7, 123, 2024, 99
- **Compared models:** No Dropout, Standard Dropout ($p=0.3$), Random-mask, Q-guided (v2 only), GA-AD

---

## Results

### Breast Cancer Wisconsin (mean ± SD over 5 seeds)

| Model | Accuracy | Macro-F1 | Test loss | Train–test gap |
|---|---|---|---|---|
| No dropout | 0.9649 ± 0.0088 | 0.9621 ± 0.0095 | 0.1325 ± 0.0332 | 0.0316 ± 0.0114 |
| Standard dropout | **0.9789 ± 0.0048** | **0.9772 ± 0.0053** | 0.0914 ± 0.0138 | 0.0152 ± 0.0063 |
| GA-AD (v1) | 0.9772 ± 0.0118 | 0.9752 ± 0.0129 | **0.0833 ± 0.0086** | **0.0099 ± 0.0089** |
| Random-mask | 0.9737 ± 0.0062 | 0.9715 ± 0.0067 | 0.0874 ± 0.0150 | 0.0163 ± 0.0057 |

GA-AD vs. standard dropout: $\Delta_{\text{acc}} = -0.0018$, 95% CI $[-0.0197,\ 0.0162]$, paired $p = 0.799$.

### MNIST (single run, seed 42, $n = 6300$)

| Model | Accuracy | Precision | Recall | F1 | Errors |
|---|---|---|---|---|---|
| No dropout | 0.9771 | 0.9771 | 0.9767 | 0.9768 | 144 |
| Standard dropout | **0.9775** | **0.9774** | **0.9772** | **0.9773** | 142 |
| Random-mask | 0.9705 | 0.9706 | 0.9702 | 0.9703 | 186 |
| Q-guided | 0.9706 | 0.9703 | 0.9705 | 0.9703 | 185 |
| GA-AD (v2) | 0.9754 | 0.9755 | 0.9748 | 0.9751 | 155 |

### Fashion-MNIST (single run, seed 42, $n = 10000$)

| Model | Accuracy | Precision | Recall | F1 | Loss |
|---|---|---|---|---|---|
| No dropout | 0.8901 | 0.8906 | 0.8901 | 0.8892 | 0.3294 |
| Standard dropout | **0.8944** | 0.8945 | **0.8944** | **0.8931** | **0.2921** |
| Random-mask | 0.8857 | 0.8851 | 0.8857 | 0.8848 | 0.3106 |
| Q-guided | 0.8926 | 0.8928 | 0.8926 | 0.8918 | 0.3150 |
| GA-AD (v2) | 0.8936 | 0.8945 | 0.8936 | 0.8929 | 0.3155 |

All pairwise 95% Wilson intervals overlap on the image tasks; they capture test-set sampling noise only, not seed variance.

---

## Limitations

1. **Single seed** on MNIST and Fashion-MNIST; seed variance is unmeasured.
2. **Unmatched training budgets:** image custom trainers lack the learning-rate-reduction rule, and MNIST custom runs are capped at 25 epochs vs. 40 for baselines.
3. **Two importance definitions** (v1 vs. v2), so results are not directly comparable across datasets.
4. **One fixed split and a small test set** on Breast Cancer (114 samples; one error = 0.88 points).
5. **No tuning or ablations** of $p$, $\alpha, \beta$ or the GA parameters.
6. **Mask granularity:** standard dropout is element-wise; GA-AD and structured controls share a mask per minibatch.
7. **Hardware differs** across runs, so timings are only indicative.

A claim of benefit would require ≥10 seeds with repeated splits, matched budgets, reverse-$Q$ and score-definition ablations, GA vs. single-best-mask vs. Q-guided at matched compute, and an outer-loop search over $p$.

---

## Repository Structure

```
.
├── notebooks/
│   ├── breast-cancer-wisconsin-diagnostic.ipynb
│   ├── mnist-digit-recognizer-ga-ad.ipynb
│   └── fashion-mnist-ga-ad.ipynb
├── paper/
│   ├── GA-AD_Dropout_IEEE_Paper.pdf
│   └── GA-AD_Dropout_IEEE_Paper.tex
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install tensorflow==2.21.0 keras==3.13.2 numpy pandas scipy scikit-learn matplotlib seaborn jupyter
```

## Usage

1. Download the datasets (Kaggle copies were used):
   - [Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
   - [MNIST Digit Recognizer](https://www.kaggle.com/c/digit-recognizer)
   - [Fashion-MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
2. Edit the data path variables at the top of each notebook (they point to `/kaggle/input/...`).
3. Run:

```bash
jupyter notebook notebooks/
```

Key hyper-parameters (`FIXED_P`, `SEEDS`, `EPOCHS`, `GA_POP`, `GA_GENS`) are defined in the first code cells.

---

## Citation

```bibtex
@misc{amir2026gaad,
  author = {Amir, Sakib},
  title  = {Importance-Guided Genetic Neuron Masking as an Alternative to Dropout:
            A Controlled Empirical Study on Tabular and Image Benchmarks},
  year   = {2026},
  note   = {BRAC University}
}
```

## License

Add a license of your choice (e.g. MIT) in a `LICENSE` file.
