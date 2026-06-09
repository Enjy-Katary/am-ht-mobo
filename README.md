# am-ht-mobo

**Multi-objective Bayesian optimization for heat treatment process parameters.** This framework uses Gaussian Process models to optimize Ultimate Tensile Strength (UTS) and Energy Consumption (Ec) simultaneously, identifying Pareto-optimal heat treatment conditions through interactive experimentation.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Expected Outputs](#expected-outputs)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)


## Features

- **Multi-objective Bayesian Optimization**: Simultaneous optimization of UTS and energy consumption
- **Interactive Workflow**: Guides users through experiment suggestions
- **Pareto Front Analysis**: Identifies and visualizes trade-offs between objectives
- **Hypervolume Tracking**: Monitors optimization progress over iterations
- **Automatic Energy Calculation**: Computes energy consumption from process parameters
- **Comprehensive Visualization**: Generates Pareto fronts, hypervolume progress, and parameter distribution plots

## Installation

### Option 1: Pip Environment (Recommended for simplicity)

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/Enjy-Katary/am-ht-mobo.git
cd am-ht-mobo

# 2. Create virtual environment
python -m venv myenv

# 3. Activate environment
# Windows:
myenv\Scripts\activate
# macOS/Linux:
source myenv/bin/activate

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt
```

### Option 2: Conda Environment

**Requirements:** Miniconda3 or Anaconda

**Step 1: Install Miniconda3** (if not already installed)
1. Download from: https://docs.conda.io/en/latest/miniconda.html
2. Run installer and follow prompts
3. **Restart your terminal** after installation

**Step 2: Setup Environment**
```bash
# 1. Clone the repository
git clone https://github.com/Enjy-Katary/am-ht-mobo.git
cd am-ht-mobo

# 2. Create conda environment from file
conda env create -f environment.yml

# 3. Activate environment
conda activate am-ht-mobo
```

## Getting Started

### 1. Run Interactive Optimization

```bash
python -m src.am_ht_mobo.bo
```

The interactive optimization will:
1. Load initial experiments from configuration
2. Train Gaussian Process models on existing data
3. Suggest next optimal experiment parameters
4. Prompt you to enter experimental results (UTS measurement)
5. Automatically calculate energy consumption
6. Generate iteration plots (Pareto front, hypervolume progress)
7. Repeat until you choose to stop

### 2. Generate Final Visualizations

After completing the optimization, run the plotting module to generate comprehensive analysis:

```bash
python -m src.am_ht_mobo.plotting
```

This generates:
- Comprehensive visualization plots
- Performance analysis

### 3. Run Demo Notebook (Optional)

```bash
jupyter notebook
# Open: notebook/00_demo_end_to_end.ipynb
```

The demo notebook provides a complete walkthrough of the optimization process with example data.

## Expected Outputs

### Generated Files

**Database:**
- `data/demo/database_demo.csv` - All experiments with parameters and results

**Pareto Points:**
- `data/pareto_optimal_points.csv` - Current Pareto-optimal solutions

**Figures (in `figures/` directory):**
- `pareto_front_iter_N.png` - Pareto front at iteration N
- `hypervolume_progress_iter_N.png` - Hypervolume improvement over iterations
- `experiment_distribution.png` - Parameter space exploration visualization

### Visualization Examples

**Pareto Front Plot:**
- X-axis: Ultimate Tensile Strength (MPa)
- Y-axis: 1/Energy (1/kWh) - maximizing efficiency
- Red points: Pareto-optimal solutions
- Blue points: Initial experiments
- Gray points: All other experiments

**Hypervolume Progress:**
- Shows optimization improvement over iterations

**Parameter Distribution:**
- 4 subplots showing exploration of parameter space
- Highlights Pareto-optimal points in each dimension

## Project Structure

```
am-ht-mobo/
├── src/
│   └── am_ht_mobo/
│       ├── __init__.py
│       ├── bo.py                    # Main Bayesian optimization logic
│       ├── energy.py                # Energy consumption calculations
│       ├── filemanager.py           # Path and configuration management
│       └── plotting.py              # Visualization functions
├── notebook/
│   ├── 00_demo_end_to_end.ipynb    # Complete demo walkthrough
│   └── notebook_filemanager.py     # Notebook-specific utilities
├── configs/
│   └── config.example.json         # Configuration template
├── data/
│   ├── demo/
│   │   └── database_demo.csv       # Experiment database
│   └── pareto_optimal_points.csv   # Pareto solutions
├── figures/                         # Generated plots
├── environment.yml                  # Conda environment specification
├── requirements.txt                 # Pip requirements
└── README.md                        
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this project in your research, please cite the following paper:


```bibtex
@article{am-ht-mobo_2026,
author = {E. Katary, K. Sekuła, D. Krok, M. Posmyk, M. Dusza, W. Matusik, K. Gruber},
title = {Heat treatment design for additively manufactured nickel superalloy via Bayesian optimization of ultimate tensile strength and energy consumption},
journal = {Virtual and Physical Prototyping},
year = {2026},
publisher = {Taylor \& Francis},
doi = {https://doi.org/10.1080/17452759.2026.2686064},
}

```
---

**Built with:** PyTorch, BoTorch, GPyTorch, Polars, Matplotlib
