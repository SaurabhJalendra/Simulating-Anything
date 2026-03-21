# Lessons Learned — Simulating Anything

## Session Efficiency
- Use `| tail -5` on all long commands to save context
- Never poll TaskOutput more than twice — use block=true
- Batch commits (fewer, larger) instead of many small ones
- Use subagents for heavy exploration

## Discovery Quality
- Depth > breadth: 1 genuine discovery > 1000 template domains
- Always validate bifurcations across 5 seeds before claiming
- SINDy polynomial fits are NOT physics discovery — they degrade on extrapolation
- Template-generated domains (same ODE, different coefficients) don't prove universality

## Technical Gotchas
- WSL command: `wsl.exe -d Ubuntu -- bash -lc "..."` (never `-e bash -c`)
- PySINDy: no model.differentiate(), use np.gradient() + x_dot param
- RSSM loading: WorldModelTrainer tree, not standalone RSSM(obs_shape=...)
- SINDy threshold: 0.005 default, lower (0.0001) for slow dynamics
- Never run 2 JAX training processes on same GPU
- Sim class names vary: check actual class name before using

## Project Strategy
- The end goal is discoveries, not metrics
- Always monitor background tasks — never launch and forget
- User has unlimited Claude usage — maximize work per session
- User wants full autonomy — don't stop, don't ask, just execute
- After every correction: update this file
