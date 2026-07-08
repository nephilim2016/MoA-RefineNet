# Adaptive Multi-Task Mixture-of-Adapters for GPR-Based Infrastructure Inspection under Diverse Electromagnetic Interference Conditions

Official implementation of the paper:

> **Adaptive Multi-Task Mixture-of-Adapters for GPR-Based Infrastructure Inspection under Diverse Electromagnetic Interference Conditions**

---

## Overview

This repository provides the implementation of the proposed **Mixture-of-Adapters (MoA)** framework for multi-task Ground Penetrating Radar (GPR) clutter suppression under heterogeneous electromagnetic interference conditions.

The proposed framework is designed to suppress three representative types of electromagnetic interference:

- **Shallow metallic diffraction and clutter**
- **Heterogeneous-medium scattering**
- **Multiple scattering and electromagnetic shielding**

---

## Repository Structure

```text
.
├── 00_original_MoA_refineNet/
│   └── Original implementation of the proposed MoA-RefineNet
│
├── 01_constraint_optimization_strategy/
│   └── Stability-aware routing optimization strategy
│
├── 02_different_loss_strategy/
│   └── Comparison of different loss functions
│
├── 03_single_refineNet/
│   └── Conventional RefineNet baseline
│
├── 04_original_refineNet_AdamW/
│   └── RefineNet trained using AdamW
│
└── 05_original_refineNet_Adam/
    └── RefineNet trained using Adam
```

---

## Main Files

Each implementation contains the following core modules.

| File | Description |
|------|-------------|
| `RefineNet.py` | RefineNet backbone network |
| `Mixture_of_Adapters.py` | Proposed Mixture-of-Adapters (MoA) module |
| `training_func.py` | Three-stage training pipeline |
| `display_results.py` | Reconstruction and visualization |
| `visualize_router_weights.py` | Routing weight visualization |
| `loss_func.py` | Loss function definitions |
| `Normalization.py` | Data normalization |

---

## Requirements

Recommended environment:

```text
Python >= 3.10
TensorFlow >= 2.15
NumPy
SciPy
Matplotlib
```

GPU acceleration is recommended.

---

## Training

The proposed framework follows the three-stage training strategy described in the paper:

1. Backbone pre-training
2. MoA routing optimization
3. Joint fine-tuning

Please refer to:

```text
training_func.py
```

for the complete training implementation.

---

## Inference

After training, reconstruction results can be generated using:

```bash
python display_results.py
```

Routing weight visualization can be generated using:

```bash
python visualize_router_weights.py
```

---

## Datasets

The synthetic datasets used in this work were generated using the forward modeling strategy described in the manuscript.

Since portions of the experimental datasets involve ongoing engineering projects and collaborative research, they are **not included** in this repository.

Users may substitute their own GPR datasets following the input format described in the paper.

---

## Reproducibility

This repository provides the complete implementation of the proposed MoA-RefineNet framework, including:

- Network architecture
- Training pipeline
- Inference pipeline
- Routing strategy
- Visualization scripts
- Experimental configurations

The implementation is intended to facilitate the reproduction and further development of the proposed methodology.

---

## License

This project is released for MIT License.
