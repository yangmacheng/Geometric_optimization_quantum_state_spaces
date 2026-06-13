# Code for "Geometric Optimization for Tight Entropic Uncertainty Relations"

[![arXiv](https://img.shields.io/badge/arXiv-2602.00595-b31b1b.svg)](https://arxiv.org/abs/2602.00595)
[![Language](https://img.shields.io/badge/Language-PYTHON-orange.svg)]()

**Authors:** Ma-Cheng Yang and Cong-Feng Qiao

This repository contains the source code associated with the research article:
> **Geometric Optimization for Tight Entropic Uncertainty Relations**  
> Ma-Cheng Yang and Cong-Feng Qiao  
> *arXiv preprint arXiv:2602.00595 [quant-ph]* (2026).  
> [View on arXiv](https://arxiv.org/abs/2602.00595)


## 📋 Requirements

The codebase is developed in **Python** and requires packages: numpy, scipy, matplotlib, pypoman, itertools, time, qutip, qbsim.


## 📂 Repository Structure


- **[`entropy_min_outerapproximate`](https://github.com/yangmacheng/Optimal_bound_entropy_UR/blob/master/entropy_min_outerapproximate.py)**
  - **Core Algorithm**: Calculates the minimal entropy of a POVM via Support-Function Based Outer Approximation.

- **[`EUR_two_measurement`](https://github.com/yangmacheng/Optimal_bound_entropy_UR/blob/master/EUR_two_measurement.py)**, **[`EUR_three_measurement`](https://github.com/yangmacheng/Optimal_bound_entropy_UR/blob/master/EUR_three_measurement.py)**
  - Two examples with two and three measurement settings for qutrit.

- **[`vertex_track_povm_entropy`](https://github.com/yangmacheng/Optimal_bound_entropy_UR/blob/master/vertex_track_povm_entropy.py)**
  - Visualization of the convergence of the outer-approximating polytope.


- **[`EUR_steering_detection`](https://github.com/yangmacheng/Optimal_bound_entropy_UR/blob/master/EUR_steering_detection.py)**
  - Quantum steering detection based on entropic uncertainty relations and visualization.



