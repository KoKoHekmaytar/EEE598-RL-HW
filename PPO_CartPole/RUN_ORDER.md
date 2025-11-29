# Execution Order for PPO CartPole Assignment

## Required Execution Order

### 1. Train Models (Seeds 0, 1, 2)
```bash
python train.py
```
**What it does:**
- Trains PPO on cart-pole balance task using seeds 0, 1, 2
- Saves trained models to `results/models/`
- Logs training metrics to `results/training_logs/`

**Output files:**
- `results/models/ppo_cartpole_seed0.zip`
- `results/models/ppo_cartpole_seed1.zip`
- `results/models/ppo_cartpole_seed2.zip`
- `results/training_logs/training_seed{0,1,2}.json`
- `results/training_logs/monitor_seed{0,1,2}.csv`

**Time:** ~30-45 minutes total (for all 3 seeds)

---

### 2. Evaluate Models (Seed 10)
```bash
python evaluate.py
```
**What it does:**
- Evaluates all trained models on seed 10
- Runs 10 episodes per model
- Logs evaluation metrics

**Output files:**
- `results/evaluation_logs/evaluation_all_seed10.json`

**Time:** ~1-2 minutes

**Prerequisites:** Must run Step 1 first (needs trained models)

---

### 3. Generate Learning Curves
```bash
python plot_learning_curves.py
```
**What it does:**
- Generates learning curve plots with mean ± std
- Shows both training and evaluation results

**Output files:**
- `results/learning_curves.png` (separate training/eval plots)
- `results/learning_curves_combined.png` (combined plot)

**Time:** ~10 seconds

**Prerequisites:** Must run Steps 1 and 2 first (needs training and evaluation logs)

---

## Complete Workflow

```bash
# Step 1: Train (takes longest)
python train.py

# Step 2: Evaluate (quick)
python evaluate.py

# Step 3: Plot (instant)
python plot_learning_curves.py
```

## Verification

After running all steps, you should have:
- ✅ 3 trained models in `results/models/`
- ✅ Training logs in `results/training_logs/`
- ✅ Evaluation logs in `results/evaluation_logs/`
- ✅ Learning curve plots in `results/`

## Troubleshooting

- **If Step 2 fails:** Make sure Step 1 completed successfully and models exist
- **If Step 3 fails:** Make sure Steps 1 and 2 completed and log files exist
- **If training is slow:** This is normal - PPO training takes time. You can reduce `total_timesteps` in `train.py` for faster testing (but lower performance)

