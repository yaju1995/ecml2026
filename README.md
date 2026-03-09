# ECML2026

## Description

This repository contains the implementation and experiments associated with the article:

**Intraday Rescheduling in Adversarial Combinatorial Bandits: Application to Decentralized EV Fleet Charging**

The project provides code to reproduce the experiment presented in the paper and includes tutorial notebooks to help users understand and experiment with the framework. The implementation relies on a preexisting simulation framework, which will be released after the review process for anonymization purposes.

---

## Requirements

- Python 3.9.21
- Conda (recommended for environment management)

---

## Set up the environment

Create a virtual environment:

```bash
conda create -n IRS-ECML2026 python=3.9.21
```

Activate the environment:

```bash
conda activate IRS-ECML2026
```

Clone the repository:

```bash
git clone https://gitlab.com/49woodframe20/ecml2026.git
```

Go to the project directory:

```bash
cd ecml2026
```

Install the dependencies:

```bash
pip install -e .
```

To reproduce the simulations from the paper:

1. Navigate to the `simulation` folder.
2. Open the `simulation_ECML2026.ipynb` notebook and run the cells to produce the results presented in the article or run the python script `simulation_ECML2026.py`.
3. Follow the  the `results_analysis` notebook to reproduces the figures of the article.