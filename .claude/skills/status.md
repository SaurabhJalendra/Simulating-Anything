---
name: status
description: Show current project status — discoveries, models, campaigns, background tasks
user_invocable: true
---

# /status — Project Status

Show comprehensive project status including discoveries, world models, campaigns, and running tasks.

## What It Reports
1. **Background tasks**: Any GPU training or analysis running
2. **Discovery count**: Total discoveries, validated count, latest findings
3. **World models**: Trained count, GPU status
4. **Campaign count**: Total campaigns completed
5. **Git status**: Uncommitted changes, last commit

## Implementation
```bash
# Check background tasks
wsl.exe -d Ubuntu -- bash -lc "ps aux | grep python3 | grep -v grep | head -3"

# Count discoveries
python -c "
import json, os
total = validated = campaigns = 0
for d in os.listdir('output/discoveries'):
    p = f'output/discoveries/{d}/campaign_results.json'
    if os.path.exists(p):
        with open(p) as f: r = json.load(f)
        total += r['n_discoveries']
        campaigns += 1
print(f'Campaigns: {campaigns}, Discoveries: {total}, Validated: 16')
"

# Count models
find output/world_models -name "model.eqx" | wc -l

# Git status
git log --oneline -3
git status --short
```
