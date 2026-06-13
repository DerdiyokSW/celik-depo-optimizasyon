# Steel Coil Warehouse Placement Optimization

> An end-to-end decision support system for heavy-industry steel warehouses that minimizes
> **rehandling** (unnecessary crane moves) and crane travel distance through predictive machine
> learning and reinforcement learning (PPO) — with a live 3D digital twin dashboard.

This project was developed as a **Computer Engineering undergraduate thesis** (BSM 498, Sakarya
University, 2025–2026) but is scoped and architected for production-grade industrial deployment
in facilities similar to Borçelik Gemlik.

---

## Problem

Steel coils (multi-ton) are stacked in a warehouse and later dispatched to production lines or
outbound transport. Two dominant costs:

- **Rehandling:** moving a coil on top to reach the one beneath — a fully unproductive crane cycle.
- **Crane distance:** total travel path per shift (energy + cycle time).

Both costs are determined almost entirely by the **placement decision** made when a coil arrives.
The decision is hard: it is **sequential** (each placement affects future costs), **stochastic**
(dispatch time, vehicle delay, priority are uncertain), and **combinatorial** (576 slots with
stack and logistics-line constraints).

---

## Architecture

The system is composed of four headless layers. Visualization is a thin shell on top.

```
[Synthetic Data Generator]
          │
          ▼
[Simulation Core] ◄── [Poisson Event Generator]
  8×36×2 = 576 slots, Chebyshev crane distance,
  stack continuity, logistics-line affinity
          │
          ▼
[Placement Policies]  ── common PlacementPolicy interface
  ├── RandomPolicy       (lower bound)
  ├── HeuristicPolicy    (rule-based, urgency-driven)
  ├── MLHeuristicPolicy  (heuristic + delay prediction)
  └── PPOPolicy          (MaskablePPO, learned)
          ▲
          │
[Delay Prediction ML]  LightGBM, MAE ≈ 6.95 min
  feeds BOTH ML-Heuristic urgency AND PPO observation
```

**Hybrid design:** the delay model stamps each vehicle with a predicted delay; placement
policies use this signal to position coils at the correct tier and slot. PPO learns this decision
end-to-end via trial and error.

---

## Key Results

Evaluation on **30 held-out scenarios** (unseen seeds), paired runs, Wilcoxon signed-rank test:

| Configuration | Best policy | Metric | Result |
|---|---|---|---|
| 2-tier + affinity, 12 ev/h | Heuristic | Rehandling | 8.57 (PPO: 10.40, p=0.11) |
| 2-tier + affinity, 12 ev/h | Heuristic | Crane distance | 26,525 m (PPO: 35,675, p<0.001) |
| Single-tier / pure routing | **PPO** | Crane distance | 26,650 m vs 28,143 (27/30, p<0.001) |
| Low load (4 ev/h) | **PPO** | Rehandling | **0.90** (lowest of all policies) |

**Main finding:** there is no universally best method — the right policy depends on warehouse
structure (stacking depth, constraints) and load regime. This motivates a **hybrid deployment**
strategy: PPO at low load, heuristic at high load.

Two self-corrections during development (overfitting discovery: in-sample 3.40 → held-out 21.62
→ pool-trained 4.13; constraint relaxation artifact → 10.40 under real constraints) demonstrate
methodological rigor.

---

## Technology Stack

| Purpose | Library |
|---|---|
| Data processing | `pandas`, `numpy`, `pyarrow` |
| Delay prediction | `lightgbm`, `scikit-learn` |
| RL environment | `gymnasium` |
| RL agent | `stable-baselines3`, `sb3-contrib` (MaskablePPO) |
| Dashboard | `plotly`, `dash` |
| Testing | `pytest` |

Python 3.11+.

---

## Project Structure

```
celik-depo-optimizasyon/
├── src/
│   ├── domain/         # data models: Coil, Vehicle, Order, ...
│   ├── data/           # synthetic data generator
│   ├── simulation/     # discrete-event simulator + Poisson event engine
│   ├── ml/             # LightGBM delay prediction model
│   ├── policies/       # PlacementPolicy interface + 4 implementations
│   ├── rl/             # gymnasium env + MaskablePPO training
│   ├── dashboard/      # Plotly/Dash 3D digital twin
│   └── evaluation/     # paired comparison pipeline
├── tests/              # pytest mirror of src/
├── docs/               # phase-by-phase specifications (Turkish)
├── runs/evaluation/    # experiment results, figures, CSVs
├── data/               # generated datasets (gitignored, reproducible)
└── models/             # trained models (gitignored, reproducible)
```

---

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Usage (in order)

```bash
# 1. Generate synthetic dataset
python -m src.data.generate_all

# 2. Train the delay prediction model
python -m src.ml.train

# 3. Launch the live 3D dashboard (uses heuristic by default)
python -m src.dashboard.app

# 4. Train the PPO agent (GPU recommended; ~3M steps)
python -m src.rl.train_ppo

# 5. Run held-out evaluation (all 4 policies)
python -m src.evaluation.compare
```

The dashboard at `http://localhost:8050` shows a live 3D warehouse view with coils colored by
urgency, side-by-side policy comparison, and a scenario selector (2-tier ↔ single-tier with
automatic PPO model swap).

---

## Tests

```bash
pytest
```

Seven test layers: determinism, data contract, physics/constraints, mask validity, policy
interface, ML pipeline, integration.

---

## Coding Conventions

- **Identifiers:** English + ASCII
- **Comments and docstrings:** Turkish (dense, explaining the *why*)
- **Type hints:** mandatory on all function signatures
- **Determinism:** every random component accepts a `seed` parameter
- **No fake metrics:** all numbers are real outputs of the codebase

---

## Citation / Acknowledgements

If you use this codebase, please cite:

```
M. Yusuf Derdiyok, "Steel Coil Warehouse Placement Optimization via Delay Prediction
and Reinforcement Learning," BSM 498 Thesis, Sakarya University, 2026.
```

The discrete-event simulation design and SLAP problem framing benefited from the open-source
warehouse simulation work by [j4n1k](https://github.com/j4n1k).

PPO methodology: Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347.
