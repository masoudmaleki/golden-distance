# Golden Distance — Python Implementation

> **Paper:** Melek M. & Melek N. (2025).  
> *Golden Distance: A New and Comprehensive Metric Definition Study  
> Facilitating Classification Performance Evaluations.*  
> Iranian Journal of Science and Technology, Transactions of Electrical Engineering.  
> https://doi.org/10.1007/s40998-025-00870-x

## What it does

Computes a single composite score (**GD**) that summarises N classification 
metrics (N = 3–10) into one number — the Euclidean distance from an ideal 
classifier in NAV–DCO space. **Lower GD = better classifier.**

Produces:
- Radar charts (performance polygon vs ideal)
- NAV–DCO scatter plots with GD annotation
- Ranked bar chart with result table

## Requirements

```bash
pip install numpy matplotlib
```

## Quick start

```python
from golden_distance import run_golden_distance

systems = [
    [0.9444, 0.9537, 0.9533, 0.9488],   # Classifier A
    [0.9259, 0.8889, 0.8929, 0.9091],   # Classifier B
]
run_golden_distance(
    systems,
    metric_names=["ACC", "Sensitivity", "Specificity", "F1"],
    system_names=["Method A", "Method B"],
)
```

## Technical notes

| N (metrics) | Method | Permutations |
|---|---|---|
| 3 – 8 | Exact enumeration | Up to 40 320 |
| 9 – 10 | Monte Carlo (default 100 000 draws) | Sampled |

The mean NAV has an exact closed form: **Av-NAV = μ² − σ²/(N−1)**,
computed alongside the numeric average as a consistency check.

## Cite this work

```bibtex
@article{melek2025golden,
  author  = {Melek, Mesut and Melek, Negin},
  title   = {Golden Distance: A New and Comprehensive Metric Definition
             Study Facilitating Classification Performance Evaluations},
  journal = {Iranian Journal of Science and Technology,
             Transactions of Electrical Engineering},
  year    = {2025},
  doi     = {10.1007/s40998-025-00870-x}
}
```
