---
name: debug
description: Systematic debugging — reproduce, isolate, root-cause, fix, verify
allowed-tools: [Read, Edit, Bash, Grep, Glob]
---

# Systematic Debugging

## Steps
1. **Reproduce**: Get the exact error (run the failing command/test)
2. **Isolate**: Find the failing file and line (read error traces)
3. **Root cause**: Understand WHY it fails (read the code, trace the logic)
4. **Fix**: Make the minimal change that fixes the root cause
5. **Verify**: Run the original failing command — confirm it passes
6. **Regression**: Run the full test suite — confirm nothing else broke

## Rules
- Fix the root cause, not the symptom
- Don't add try/catch to hide errors
- Don't skip tests with .skip or @pytest.mark.skip
- If unsure, add a temporary log and re-run before guessing
- For JAX/GPU issues: always run via WSL2 (`wsl.exe -d Ubuntu -- bash -lc "..."`)
- For SINDy/PySR: check known gotchas in CLAUDE.md Section 10 first
