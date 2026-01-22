# Neural Superstatistics for Bayesian Estimation of Time-Varying Parameters

This repository contains code, tools, and examples associated with **Neural Superstatistics**, a method for Bayesian inference in dynamic cognitive modeling introduced in *Schumacher et al.* (2023). ([nature.com](https://www.nature.com/articles/s41598-023-40278-3))

## 📘 Overview

Traditional cognitive models often assume that model parameters are **static** (unchanging over time). However, many cognitive processes are inherently **dynamic**, with parameters that evolve across time. Modeling these temporal dynamics can reveal richer structure and improve inferential accuracy. ([nature.com](https://www.nature.com/articles/s41598-023-40278-3))

**Neural superstatistics**:
- Frames cognitive models as hierarchical systems with *observation-level* and *parameter-dynamics* components.
- Uses **amortized Bayesian inference** to perform efficient parameter estimation.
- Enables recovery of both *time-varying* and *time-invariant* parameters from data. 

The method is benchmarked against existing frameworks and applied to dynamic versions of cognitive models, demonstrating superior efficiency in capturing temporal structure in behavioral data. ([nature.com](https://www.nature.com/articles/s41598-023-40278-3))

## 🚀 Key Features

- **Dynamic Bayesian Inference:** Moves beyond static IID parameter assumptions to capture *temporal evolution* of latent model parameters.
- **Amortized Deep Learning:** Trains neural networks on simulated data once, then reuses them for fast inference across multiple datasets.
- **Broad Applicability:** Suitable for complex cognitive models where traditional Bayesian estimation is computationally expensive or infeasible.


## 📄 Citation

If you use this repository in your research, please cite:

> Schumacher L., Bürkner P.-C., Voss A., Köthe U., & Radev S. T. (2023). *Neural superstatistics for Bayesian estimation of dynamic cognitive models*. *Scientific Reports* 13:13778.
