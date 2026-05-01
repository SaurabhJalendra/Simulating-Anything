---
name: review
description: Reviews code changes for bugs, security, quality, and project convention adherence
allowed-tools: [Read, Grep, Glob, Bash]
---

# Code Review

## Steps
1. Get the diff: `git diff` (unstaged) or `git diff --cached` (staged)
2. Review for:
   - **Bugs**: Logic errors, off-by-one, null/undefined access
   - **Security**: Injection, auth bypass, data exposure
   - **Quality**: Naming, complexity, duplication
   - **Conventions**: Does it follow project patterns? (`from __future__ import annotations`, type hints, Google docstrings, no emojis)
   - **Tests**: Are changes tested?
3. Check types: `ruff check src/ tests/`
4. Check known gotchas in CLAUDE.md Section 10

## Output Format
For each issue:
- CRITICAL / WARNING / SUGGESTION
- File:line — what's wrong — how to fix

If no issues: "Code looks good. No issues found."
