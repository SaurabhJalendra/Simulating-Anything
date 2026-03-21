---
name: demo
description: Run end-to-end autonomous demo — NL question to discovered equation
user_invocable: true
---

# /demo — End-to-End Autonomous Demo

Demonstrate the full pipeline: natural language question → generated simulation → discovered equation.

## Usage
```
/demo "How does spring constant affect oscillation period?"
/demo "How does population density affect disease spread?"
/demo "How does temperature affect chemical reaction rate?"
```

## What It Does
1. SimulationGeneratorAgent generates Python simulation from NL (Claude CLI)
2. SimulationValidator runs 7 automated checks
3. Parameter sweep across the key variable (30 points)
4. Observable extraction (period, amplitude, peak)
5. Fit scaling law: observable = f(parameter)
6. Compare to known theoretical result if available
7. Report: generated code, validation, discovered equation, correlation

## Requirements
- Claude Code CLI must be installed in WSL (`which claude`)
- WSL2 with Python venv activated

## Previous Demo Results
- "Spring constant vs period": T ~ 1/sqrt(k), correlation = 0.986
