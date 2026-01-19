# Automated Generation of RCS and ISAR Datasets

**Paper Title:** Automated Generation of RCS and ISAR Datasets for Macro and Micro Scale Aircraft Recognition  
**Authors:** Diego Paramés, David Cabornero, Iván González, Lorena Lozano, Felipe Cátedra  
**Affiliation:** University of Alcalá, Madrid, Spain

## Overview

This software provides an automated pipeline for generating massive, labeled synthetic radar datasets (RCS matrices and ISAR images) for training Deep Learning models (specifically CNNs) for Automatic Target Recognition (ATR).

The tool manages the entire workflow:
1.  **Geometry Management:** Processing STL files (with facet reduction capabilities via Blender).
2.  **Physics-Based Simulation:** Calculating RCS using electromagnetic solvers.
3.  **Dataset Construction:** Formatting output into labeled datasets with traceability, ready for neural network training.

**Key Features:**
* **Modular Design:** Decoupled from the EM solver. Includes **OpenRCS** for immediate reproducibility but supports high-fidelity solvers (e.g., GEMIS, newFASANT, CADFEKO).
* **Automated Labeling:** Generates CSV labels and folder structures automatically.
* **Data Augmentation:** Built-in tools for angular constraint filtering and Signal-to-Noise Ratio (SNR) injection.
* **Model Agnostic:** Outputs standard `.npy` matrices and `.png` images compatible with CNNs, Transformers, or RNNs.

## Requirements

The software requires **Python 3.10+** and **Blender 4.2** (for geometry operations).

### Python Dependencies
Install the required libraries using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

*Key dependencies include:*
* `numpy` (tested on 2.2.6)
* `numpy-stl` (tested on 3.2.0)
* `pandas` (tested on 2.3.3)
* `matplotlib` (tested on 3.10.3)

## Usage & Reproducibility

The software is controlled via the Command Line Interface (CLI) `generate.py`.

### 1. Basic Usage (Reproducible Example)
To validate the software functionality using the included OpenRCS solver and the sample "Tank" geometry:

**Step A: Generate RCS Data**
This command generates RCS matrices for a Tank geometry with a specific angular sweep.
```bash
python generate.py rcs --geometries Tank --case 0 --samples 3 --theta 50 130 --phi -40 40
```

**Step B: Create Labeled Dataset (ISAR)**
This command processes the raw RCS data into ISAR images with noise injection (15 dB SNR) and creates the final labeled dataset.
```bash
python generate.py label --geometries Tank --case 0 --samples 3 --pov front --cw 40 --snr 15 --data ISAR
```

### 2. Advanced Usage
For generating data for complex aircraft (e.g., Embraer Phenom, Rafale, Yak-130) as described in the manuscript:

```bash
# Generate Raw RCS
python generate.py rcs --geometries Embraer_Phenom_100 Yakovlev_Yak-130 Dassault_Rafale --case 0 --samples 50 --theta 40 140 --phi -50 50

# Generate Labeled Training Set
python generate.py label --geometries Embraer_Phenom_100 Yakovlev_Yak-130 Dassault_Rafale --case 0 --samples 50 --pov front --cw 50 --snr 15 --data ISAR
```

## Configuration Arguments

| Argument | Description | Example |
| :--- | :--- | :--- |
| `--geometries` | List of folder names containing STL files | `Tank` `Dassault_Rafale` |
| `--case` | ID for radar parameters (frequency/resolution) | `0` |
| `--samples` | Number of samples to generate per geometry | `50` |
| `--theta` / `--phi` | Angular limits for generation [min max] | `--theta 40 140` |
| `--pov` | Point of View for angular constraint | `front`, `top`, `left` |
| `--cw` | Cone width (degrees) around the POV | `50` |
| `--snr` | Signal-to-Noise Ratio (dB) for noise injection | `15` |
| `--data` | Output format | `ISAR` or `RCS` (Real & Im, Amp & Ph formats)|
