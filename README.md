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

The software requires **Python 3.10+** and **Blender 4.2** (optional, only for the subcommand `fillstl`).

## Installation

### Clone the repository

To clone the repository to your local machine:

```bash
git clone https://github.com/diegopm-uah/SoftwareX_Submission
cd SoftwareX_Submission
```

### Create a virtual environment (optional)

Although this is an optional step, creating a virtual environment is strongly recommended to avoid cross-dependency issues. 

The instructions below explain how to create a virtual environment in Python. However, there are several package managers, such as uv or conda, that are equally suitable for this task.

To create a virtual environment called ``.venv``:

```bash
python -m venv .venv
```

To enter into the virtual environment, if you are in Windows PowerShell:
```bash
.venv\Scripts\activate
```

In Linux or MacOS:
```bash
source .venv/Scripts/activate
```

### Install Python Packages
Install the required packages using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

*Key dependencies include:*
* `numpy` (tested on 2.2.6)
* `numpy-stl` (tested on 3.2.0)
* `pandas` (tested on 2.3.3)
* `matplotlib` (tested on 3.10.3)


## Usage & Examples

The software is controlled via the Command Line Interface (CLI) `generate.py`. Some examples of every functionality are given below:

### Add new STL geometry

There are new geometries that would be introduced on the program (for instance, `new_geometry.stl` and `other_geometry.stl`). Firstly, they must be copied on the `data/stl` folder. Next, the subcommand `addstl` provide the logic to refresh the internal datasets.

```bash
python generate.py addstl new_geometry other_geometry --category UAV
```

In this case, two new geometries were added with the category `UAV`. Please note that this argument is optional, and can be simplified as `-c`. That is, this command is equivalent to the previous one:

```bash
python generate.py addstl new_geometry other_geometry -c UAV
```

Besides, should be noted that the number of geometries that can be introduced at the same time is not limited, but the whole set must correspond to the same category.

### Remove STL geometry of the program

As a counterpart of the previous subcommand, `rmstl` allows the user to remove a STL geometry that is currently in the program. Let's assume that `new_geometry.stl` and `other_geometry.stl` should be removed:

```bash
python generate.py rmstl new_geometry other_geometry
```

Once executed, the user cannot use the next commands with removed geometries, and they are not at the `UAV` anymore. Nonetheless, they are still on the `data/stl` folder and removing them is responsibility of the user.

### Automatic fill of STL data

After adding several STLs via command `addstl` or changing a STL file from the `data/stl` folder, this command must be executed for sanity check:

```bash
python generate.py fillstl
```

### Remesh a STL

To remesh an STL with a different number of facets, Blender 4.2+ must be installed with the Python API. The path of python must be indicated on the `BLENDER_PYTHON_PATH` variable in `generate.py`. Two equivalent examples of remeshing to 10K facets are:
```bash
python generate.py nfacets new_geometry other_geometry --num_facets 10000
python generate.py nfacets new_geometry other_geometry -n 10000
```

### Create a simulation scenario

For traceability reasons, the parameters of every simulation must be stored before being used. The task of `case` subcommand is to prepare the simulation and the set of auxiliary folders. The parameters that must be introduced are:

| Argument | Short Option | Description | Example |
| :--- | :--- | :--- | :--- |
| `theta`/`phi` | | Determines whether the sweep of the RCS is made in $\theta$ or $\phi$, fixing the other component. | `theta` |
| `--delta_angle` | `-da`| Defines the spacing between consecutive sweep angles. In an $num_{angle} \times num_{freq}$ simulation grid, this value represents the difference $(\Delta\theta)$ or $(\Delta\phi)$ between adjacent angular samples. | `--delta_angle 0.1` |
| `--delta_freq` | `-df` | Defines the spacing between adjacent frequency samples. In an $n \times n$ ISAR simulation, this value represents the interval between consecutive frequency points in the bandwidth. | `--delta_freq 0.05` |
| `--central_freq` | `-cf` | Central frequency of the ISAR simulation | `--central_freq 10e9` |
| `--num_angle` | `-na` | Number of angles per image | `--num_angle 64` |
| `--num_freq` | `-nf` | Number of frequencies per image | `--num_freq 32` |

For instance, to create ISAR images of resolution $64x32$ (being the X axis the cross-range and the Y axis the frequency-range) centered in $10 GHz$ with $(\Delta\theta)=0.1$ and $(\Delta f)=0.05$, the following command must be executed:

```bash
python generate.py case theta --delta_angle 0.1 --delta_freq 0.05 --central_freq 10e9 --num_angle 64 --num_freq 32
```

These parameters are listed on `data/rcs.csv`, and must be queried to know the identifier.

### Generate RCS Data

The subcommand `rcs` generates RCS matrices for a Tank geometry with a specific sweep (created with `case` command). For instance, given the previous configuration (listed in `data/rcs.csv` with the identifier $0$), RCS simulations are needed for the `new_geometry.stl` and the `other_geometry.stl`. To generate 300 samples with $\theta\in(50º,130º)$ and $\phi\in(-40º,40)$:

```bash
python generate.py rcs --geometries new_geometry other_geometry --case 0 --num_samples 300 --theta 50 130 --phi -40 40
```

Alternatively, instead of indicating the angular margins explicitly, if the front, bottom, left, etc. parts are the sections of interest, they can be explicitly written with the `--pov` argument. The following choices are available:

| pov | theta | phi |
| :--- | :--- | :--- |
| top | 0º | - |
| front | 90º | 0º |
| left_side | 90º | 90º |
| back | 90º | 180º |
| right_side | 90º | 270º |
| bottom | 180º | - |

Besides, a cone width argument (`--cw`)must indicate the amplitude of the cone on degrees, which will be equal for theta and phi. For instance, the previous example is centered on the front side and the cone width is of 40º in theta and phi. Thus, an equivalent command would be:

```bash
python generate.py rcs --geometries new_geometry other_geometry --case 0 --num_samples 300 --pov front --cw 40
```

### Create Labeled Dataset (ISAR)
This command processes the raw RCS data created with the subcommand `rcs` into ISAR images with noise injection (15 dB SNR) and creates the final labeled dataset. The arguments are:

| Argument | Short Option | Description | Example |
| :--- | :--- | :--- | :--- |
| `--geometries` | List of folder names containing STL files | `Tank` `Dassault_Rafale` |
| `--case` | ID for radar parameters (frequency/resolution) | `--case 0` |
| `--num_samples` | Number of samples to generate per geometry | `--samples 50` |
| `--theta` / `--phi` | Angular limits for generation [min max] | `--theta 40 140` |
| `--pov` | Point of View for angular constraint | `front` |
| `--cw` | Cone width (degrees) around the POV | `50` |
| `--SNR` | Signal-to-Noise Ratio (dB) for noise injection | `15` |
| `--data` | Output format | `ISAR`, `rcs_complex` (Real & Im) , `rcs_amp` (Only amplitude), `rcs_ph` (only phase), `rcs_amp_ph` (amplitude and phase)|


To generate randomly 3 previous raw RCS data into ISAR images with noise injection (in this case, 15 dB SNR, but noise is optional) and create the final labeled dataset:
```bash
python generate.py label --geometries new_geometry other_geometry --case 0 --num_samples 3 --pov front --cw 40 --SNR 15 --data ISAR
```

Thus, 3 cases will be found in the `CNN/labeleld_dataset/data` relative path in the format requested. The whole set of cases are listed on `CNN/labeleld_dataset/manifest.csv`.