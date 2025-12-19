# Ligand-Recptor

Enhanced Graph Neural Network models and analysis pipelines for predicting ligand–receptor (protein) binding affinity with uncertainty estimation and interpretability tools.

This repository contains research code, data-generation utilities, model implementations, training/evaluation scripts, and figure-generation utilities used to develop and analyze an Enhanced GNN architecture with Bayesian layers and Monte Carlo Dropout for uncertainty quantification.

---

## Highlights

- Enhanced GNN v2 implementation with BayesianLinear layers and Monte Carlo Dropout
- Cross-modal attention fusion between ligand, protein, and interaction feature streams
- Uncertainty-aware training and evaluation (predictive mean + uncertainty)
- Scripts for data generation, baseline comparison, interpretability, and figure generation
- Dockerfile and requirements for deployment

---

## Quick start

Requirements
- Python 3.8+ (works with 3.8, 3.9, 3.10)
- PyTorch (version compatible with your CUDA or CPU setup)
- Common scientific packages: numpy, pandas, scikit-learn, matplotlib, seaborn, tqdm

Recommended: use a virtual environment.

PowerShell example (Windows):

1) Clone the repository (already done in this workspace):
   git clone https://github.com/parker-ryan1/Ligand-Recptor

2) Create and activate a virtual environment (PowerShell):
   python -m venv .venv; .\.venv\Scripts\Activate.ps1

3) Install dependencies:
   pip install -r deployment/requirements.txt

Note: `deployment/requirements.txt` contains packages used for deployment and common experiments. Install a compatible version of PyTorch separately if you need GPU support; see https://pytorch.org for the installation command that matches your CUDA version.

---

## Typical workflows

1) Generate / load datasets
- The repository includes data-generation routines that synthesize ligand, protein and interaction features for development and experiments. See `src/enhanced_gnn_v2_uncertainty.py` and helper scripts under `src/`.

2) Train the enhanced model
- Run the main training script:
  python src/enhanced_gnn_v2_uncertainty.py

This script will generate a synthetic dataset (if not present), train the Enhanced GNN v2 with Bayesian layers, evaluate on a held-out test split, and save model + results to `results/enhanced_v2_uncertainty/`.

3) Analyze baselines and produce figures
- Baseline and comparison scripts:
  - `src/baseline_comparison.py`
  - `src/scale_to_19k_training.py` (dataset scaling experiments)
  - `src/real_data_validation.py` (validation on real/held-out data)
  - `src/generate_publication_figures.py` (figure generation pipeline)

Run these scripts with Python; inspect saved figures in the `figures/` directory.

---

## Key files and directories

- `src/` - Model implementations and analysis scripts (training, evaluation, figure generation)
  - `src/enhanced_gnn_v2_uncertainty.py` - Main enhanced model implementation with uncertainty quantification and a full training/evaluation pipeline
  - `src/baseline_comparison.py` - Baseline model experiments and comparisons
  - `src/generate_publication_figures.py` - Script to produce figures used in analysis
  - `src/*_analysis.py` - Additional analysis utilities
- `figures/` - Generated figures (PNG/PDF) for publication and analysis
- `deployment/` - `Dockerfile` and `requirements.txt` for containerized experiments
- `results/` - Created at runtime; stores model checkpoints and JSON results
- `references.bib` - Bibliography used for manuscript/notes

---

## Outputs

Running the training/evaluation scripts will create a `results/` folder containing:
- Model checkpoints (e.g. `enhanced_v2_model.pth`)
- JSON summary files with metrics and training history
- Plots saved under `figures/`

---

## Reproducibility notes

- Scripts use seeded RNGs for reproducible dataset splits (see `torch.Generator().manual_seed(42)` usage).
- Training hyperparameters (learning rate, batch size, epochs, weight decay) are set in the top-level scripts and can be edited directly or refactored to use a config file/CLI if desired.
- For large-scale experiments or GPU training, set up a matching PyTorch + CUDA environment before running.

---

## How to cite

If you use code or ideas from this repository in a publication, please cite the project and any accompanying manuscript. Bibliographic references are available in `references.bib`.

---

## Contributing

Contributions are welcome. Suggested workflow:
1. Fork the repository
2. Create a feature branch
3. Run tests (if added) and ensure reproducibility
4. Open a pull request describing your changes

Please add a LICENSE file if you intend to allow reuse; this repository does not include a license file by default.

---

## License

No license specified in this repository. Add a LICENSE file (e.g. MIT) if you want to make the code reusable.

---

If you want, I can:
- add a short example notebook that runs a quick toy training/evaluation loop
- add CLI wrappers to the main scripts for configurable runs
- add a LICENSE file (MIT/Apache) and a CITATION file

