# Macro-Consciousness Modeling: A Multi-Level Computational Framework

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16937283.svg)](https://doi.org/10.5281/zenodo.16937283)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

This repository is the official companion code for the paper:

> **Vladimir Khomyakov (2025).** _A Multi-Level Computational Model of Macro-Consciousness with Self-Organizing Inter-Cluster Networks, Predictive Adaptation, and Reproducible Python Simulations._ Zenodo. https://doi.org/10.5281/zenodo.16937283

It provides a full implementation of the computational framework described in the paper, enabling fully reproducible simulations and interactive explorations of the model.

## Abstract

This work presents a hierarchical computational framework for modeling emergent macro-consciousness. The model is based on:
*   **Micro-Level:** "Self" units that minimize entropy and adapt to their environment.
*   **Meso-Level:** Clusters of units with self-organizing interconnections and memory.
*   **Macro-Level:** A global cognitive projection that emerges from cluster dynamics and provides top-down feedback.

The framework integrates adaptive learning, predictive task alignment, and a Hebbian-like rule for self-organizing inter-cluster weights. Simulations demonstrate the emergence of macro-cognitive coherence, adaptive network reconfiguration, and the system's ability to align with external tasks.

## Repository Structure

```
macro_consciousness/
│
├── scripts/ # Main simulation scripts
│ ├── full_consciousness_interactive.py # Interactive sim with 3D network & sliders
│ ├── full_consciousness_3D.py # 3D network visualization animation
│ ├── full_consciousness_panel.py # Non-interactive run → full panel of plots
│ └── self_learning_consciousness_with_weight_dynamics.py # Core weight dynamics sim
│
├── figures/ # (Directory for saving output figures)
├── data/ # (Directory for saving output data)
├── LICENSE # MIT License
└── README.md # This file
```

## Installation & Dependencies

To run the simulations, you need Python 3.9 or later. The required libraries can be installed via pip:

**Create a virtual environment (recommended)**

```bash
python -m venv consciousness-env
# On macOS/Linux:
source consciousness-env/bin/activate
# On Windows (PowerShell):
consciousness-env\Scripts\activate
```

### Install dependencies

```bash
pip install numpy matplotlib networkx
```

## Usage
All main scripts are in the scripts/ directory. Each focuses on a different aspect of visualization.

1. **For an interactive experience (recommended first step):**

Launches a 3D network visualization with sliders to adjust key parameters (eta, alpha, beta) in real-time.

```bash
python scripts/full_consciousness_interactive.py
```

2. **To view a pre-rendered 3D animation:**

Runs a fixed simulation and animates the resulting 3D network dynamics.

```bash
python scripts/full_consciousness_3D.py
```

3. **To generate a comprehensive panel of plots:**

Runs the simulation and generates a multi-plot figure showing entropy, macro-dynamics, cluster states, task signals, weight dynamics, and correlations. This is best for analysis.

```bash
python scripts/full_consciousness_panel.py
```

4. **To run the core simulation:**

Runs the model and plots the essential dynamics (entropy, macro-projection, clusters, tasks, weights).

```bash
python scripts/self_learning_consciousness_with_weight_dynamics.py
```

## Expected Results & Outputs

Running the scripts will generate visualizations of the following emergent phenomena:

**Global Entropy Reduction:** The system minimizes its total entropy over time, displaying self-organizing behavior.

**Predictive Macro-Dynamics:** The macro-projection vector aligns with and predicts the external task signal.

**Cluster Synchronization:** Meso-level clusters show coordinated dynamics and varying degrees of correlation with the macro-state.

**Self-Learning Network Weights:** Inter-cluster weights (W) evolve based on a Hebbian-like rule, forming a complex, adaptive network.

**Spatial-Temporal Correlation:** Heatmaps and line plots showing how clusters couple with the emerging macro-consciousness.

## Citation
If you use this model or code in your research, please cite the original publication:

```bibtex
@misc{khomyakov_vladimir_2025_16937283,
  author       = {Khomyakov, Vladimir},
  title        = {A Multi-Level Computational Model of Macro-Consciousness with Self-Organizing Inter-Cluster Networks, Predictive Adaptation, and Reproducible Python Simulations},
  month        = aug,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.16937283},
  url          = {https://doi.org/10.5281/zenodo.16937283}
}
```

- **Version-specific DOI:** [10.5281/zenodo.16937283](https://doi.org/10.5281/zenodo.16937283)  
- **Concept DOI (latest version):** [10.5281/zenodo.16937282](https://doi.org/10.5281/zenodo.16937282)  
- **Download PDF:** [Direct link to paper on Zenodo](https://zenodo.org/records/16937283/files/macro_consciousness_model.pdf?download=1)

## Related Links & References

**Theoretical Foundation:** This work is based on the principles of _Subjective Physics_ by A. Kaminsky (DOI: 10.5281/zenodo.15098840).

**Predecessor Model:** Builds upon the "Minimal Model of Cognitive Projection" (DOI: 10.5281/zenodo.16888675).

**Author's Website:** https://digitalphysics.ru/

## Keywords

macro-consciousness modeling, self-organizing networks, cognitive projection, observer entropy, predictive adaptation, multi-level cognitive architecture, subjective physics, reproducible python simulations, inter-cluster coupling, information-theoretic learning

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
This work is based on principles of Subjective Physics by A. Kaminsky and the minimal model of cognitive projection.