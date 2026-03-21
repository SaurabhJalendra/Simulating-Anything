---
name: train
description: Train RSSM world models on specified domains using RTX 5090 via WSL2
user_invocable: true
---

# /train — Train World Models

Train RSSM world models on specified domains using the RTX 5090 GPU via WSL2.

## Usage
```
/train <domain1,domain2,...> [--epochs 200] [--background]
```

## Examples
```
/train lorenz,rossler,chua
/train social_epidemic --epochs 300
/train coral_reef,tumor_immune --background
```

## What It Does
1. Generates 50 training trajectories per domain
2. Trains RSSM (512 GRU + 32x32 categorical) for 200 epochs
3. Evaluates dream quality (MSE, error growth)
4. Saves checkpoint to output/world_models/<domain>/model.eqx
5. ~7 minutes per domain on RTX 5090

## Implementation
```bash
wsl.exe -d Ubuntu -- bash -lc "cd '/mnt/d/Git Repos/Simulating-Anything' && source .venv/bin/activate && python3 scripts/train_world_models_generic.py --domains <domains> --epochs 200 --n-traj 50 --seq-len 200"
```

## Important
- MUST run through WSL2 (JAX GPU doesn't work on native Windows)
- Never run 2 training processes simultaneously (GPU contention)
- Auto-discovery: unregistered domains are detected by class naming convention
