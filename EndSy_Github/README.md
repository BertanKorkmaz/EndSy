# EndSy – Agent-Based Simulation of Early Multicellularity
*A minimal framework to explore how mitochondrial “surveillance” reshapes cell‑population dynamics.*

---

## 1 Overview
**EndSy** is a Python agent‑based model (ABM) that tracks nutrient diffusion, cell damage, proliferation, and energy demand on a 2‑D grid.

Two complementary pipelines are provided:

| Pipeline              | Purpose                                                                                                   | Entry‑point                         |
|-----------------------|-----------------------------------------------------------------------------------------------------------|-------------------------------------|
| **With mitochondria** | Simulate the full scenario where each cell can be removed by a mitochondrial watchdog module.            | `EndSy.py`                          |
| **Without mitochondria** | Generate training data in a mitochondria‑free world.                                                      | `Mito_Training/EndSy_no_mito.py`    |

The training data (CSV files) are merged and used to learn a lightweight logistic‑regression classifier that mimics *mitochondrial surveillance*.  
The trained model is stored in **`mito.pkl`** and automatically loaded by **`EndSy.py`**.

---

## 2 Repository Layout
```text
EndSy_Github/
│
├── EndSy.py               # Main simulation (MITO_ENABLED True/False toggle)
├── mito.pkl               # Pre‑trained mitochondria classifier
│
└── Mito_Training/
    ├── EndSy_no_mito.py   # Simulation WITHOUT mitochondria
    ├── mito_data.py       # Parses no‑mito runs → CSV datasets
    ├── mito_trainer.py    # Merges CSVs + trains logistic model → mito.pkl
    └── mito_sim_data/     # Auto‑generated CSVs (one per run)
```

---

## 3 Setup
```bash
# 1 Create a clean environment (optional but recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scriptsctivate

# 2 Install requirements
pip install -r requirements.txt
#   numpy matplotlib pandas scikit‑learn joblib imageio ...
```
> **Python ≥ 3.9** is recommended.

---

## 4 Quick Start

### 4.1 Run the full model (with mitochondria)
```bash
python EndSy.py
```
Key flag inside **`EndSy.py`**:
```python
MITO_ENABLED = True   # → False to disable removal
```

### 4.2 Reproduce the training pipeline from scratch
```bash
# Step 1 – simulate 50 no‑mito replicates (seed 42 is baked in)
python Mito_Training/EndSy_no_mito.py

# Step 2 – convert logs to tidy CSVs
python Mito_Training/mito_data.py        --runs 50        --out_dir Mito_Training/mito_sim_data

# Step 3 – train the logistic‑regression classifier
python Mito_Training/mito_trainer.py
# → outputs mitochondria_classifier.pkl  (renamed to mito.pkl in the root)
```
Now re‑run **`EndSy.py`** and the fresh model will be picked up automatically.

---

## 5 Key Parameters (edit in `EndSy.py`)
| Variable              | Meaning                            | Default |
|-----------------------|------------------------------------|---------|
| `width`, `height`     | Grid dimensions                    | 50 × 50 |
| `capacity`            | Max nutrient per grid cell         | 5       |
| `daily_increment`     | Nutrient influx per timestep       | 0.5     |
| `SCENARIO_*`          | Scheduled nutrient pulses          | see code|
| `pro_slowdown_factor` | Penalty for overcrowded cells      | 0.99    |

---

## 6 Dependencies
- **Core**: `numpy`, `matplotlib`, `pandas`  
- **ML** : `scikit-learn`, `joblib`  
- **Media (optional)**: `imageio`

A complete list is in `requirements.txt`.

---

## 7 Citing / Acknowledgements
If you use **EndSy** in academic work, please cite our forthcoming preprint (link TBA).  
The model was inspired by discussions on mitochondrial roles in early multicellularity (Korkmaz & Kurtoglu 2025).

---

## 8 License
[MIT](LICENSE)
