# Discrete Diffusion Correctors

Code for the experiments and figures in the thesis.

## How to run

1. Run `grid_search.py` to generate the method particles (see below).
2. Run the `Producing_plots.ipynb` notebook to reproduce the figures.

The two experimental settings in the thesis correspond to the two method folders:
- **Setting A** → `methods_N30/`
- **Setting B** → `methods_N200/`

## Source files

### `main_code.py`
Core implementation. Contains:
- The **Gillespie** (exact) sampler and the **tau-leaping** sampler.
- The corrector steps: `random_masking_corrector_step`, `prism_corrector_step`,
  and `informed_corrector_step`.
- The `DiffusionSamples` class, which represents one problem instance and stores
  the forward particles together with the reverse particles produced by each method.

### `main_code_parallel.py`
The same functionality as `main_code.py`, adapted to run across multiple CPUs in parallel.

### `tests.py`
All plotting and evaluation functions used to produce the figures.

### `grid_search.py`
Creates the problem instance and runs the full grid search described in the thesis.
It creates a folder containing one `.pkl` file per method (storing that method's
particles), plus a `meta.pkl` file holding the forward particles and the setting
parameters (`N`, `L`, `r`, `mu`, etc.).

By default this produces `methods_N200/` (setting B). To produce `methods_N30/`
(setting A), change `N` from 200 to 30 and update the output directory name
to `methods_N30`, then run again.

## Notebook

### `Producing_plots.ipynb`
Reproduces the figures in the thesis, in three sections:
1. **Tests on marginal Hellinger, Gillespie, and tau-leaping** — self-contained;
   can be run on its own.
2. **Tests - N30** and **Tests - N200** — require the precomputed method folders `methods_N30/` and `methods_N200/`
   to be present (produced by `grid_search.py` as described above).
