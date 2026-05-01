---
name: docs
description: Unified documentation manager — create, update, and maintain all project docs via subcommands (changelog, sync, idea, adr, roadmap, contributing, security, troubleshooting, pr, release)
allowed-tools: [Read, Write, Edit, Bash, Grep, Glob]
---

# Docs — Unified Documentation Skill

One skill to manage ALL project documentation. Use `$ARGUMENTS` to pick the mode.

## Modes

### `/docs` or `/docs changelog` → Update CHANGELOG.md
1. Run `git log --oneline --since="today"` for today's commits
2. Run `git diff main...HEAD --stat` for changes vs main
3. Append to CHANGELOG.md under `## [Unreleased]` with sections:
   `### Added` / `### Changed` / `### Fixed` / `### Removed`
4. Use past tense, be specific, include file references

### `/docs pr` → Generate PR description
1. `git log main...HEAD --oneline` for all commits
2. `git diff main...HEAD --stat`
3. Generate:
```
## Summary
[1-3 bullets]

## Changes
- [key change + file ref]

## Testing
- [how to verify]

## Notes
[anything reviewers should know]
```

### `/docs release` → Generate user-facing release notes
1. `git describe --tags --abbrev=0` for last tag
2. `git log [last-tag]...HEAD --oneline`
3. Generate notes — audience is users, not developers. Group: New, Improved, Fixed, Breaking

### `/docs sync` → Detect and fix documentation drift
1. Read all docs: README.md, CHANGELOG.md, docs/*.md, inline docstrings
2. Check each claim against current codebase
3. Flag discrepancies (outdated features, removed endpoints, wrong setup steps)
4. Dispatch `documenter` agent to fix, or report for user decision

### `/docs idea` → Create or update IDEA.md
The project's north star — what we're building, why, for whom.
```markdown
# [Project] — Idea Document

## The Problem
## The Solution
## Who It's For
## Why This, Why Now
## Success Looks Like
## Non-Goals   <- Critical: prevents scope creep
## Open Questions
```

### `/docs adr "decision title"` → Create Architecture Decision Record
1. Check `docs/adr/` for existing ADRs, find next number (zero-padded)
2. Create `docs/adr/NNNN-decision-title.md`:
```markdown
# ADR-NNNN: [Decision Statement]
**Date:** YYYY-MM-DD
**Status:** proposed | accepted | deprecated | superseded by ADR-XXXX

## Context
## Decision
## Consequences
### Positive / ### Negative / ### Risks
## Alternatives Considered
```
Rule: NEVER edit an accepted ADR — supersede it.

### `/docs roadmap` → Create or update ROADMAP.md
```markdown
# Roadmap — Last updated: YYYY-MM-DD
## Now (2-4 weeks)
## Next (1-3 months)
## Later (exploratory)
## Recently Shipped (last 10-15)
## Not Doing   <- Critical: prevents same rejected requests coming back
```

### `/docs contributing` → Generate CONTRIBUTING.md
Detect project setup, generate:
- Development Setup (from package.json/Makefile/pyproject.toml)
- Workflow (fork → branch → test → commit → PR)
- Standards (code style, commit messages, tests required)
- PR Process
- Code of Conduct link

### `/docs security` → Generate SECURITY.md
```markdown
# Security Policy
## Reporting a Vulnerability   <- Private channel, not GitHub issue
## Supported Versions
## Security Considerations   <- Data handling, auth, deps, secrets
## Known Limitations
## Security History   <- CVEs fixed
```

### `/docs troubleshooting` → Create or update TROUBLESHOOTING.md
1. Grep codebase for common error messages
2. Read `tasks/lessons.md` for past debugging sessions
3. Generate with EXACT error messages (users search for them):
```markdown
# Troubleshooting
## Installation Issues / Runtime Issues
### Error: [exact error text]
**Cause:** [Why this happens]
**Fix:** [Specific steps]
**Prevention:** [How to avoid]
```

## Rules (apply to all modes)
- Match existing doc style in the project
- Never duplicate info across docs — link to source of truth
- Focus on WHY, not WHAT
- Use EXACT error messages in troubleshooting
- NEVER silently change docs — report what's changing and why
- Update, don't rewrite — preserve history and context
- Every command in docs must actually work (test before documenting)
- No emojis in code or documentation (per project CLAUDE.md)

## When to Use Which Mode
| Situation | Mode |
|---|---|
| After code changes | `changelog` |
| Opening PR | `pr` |
| Tagging release | `release` |
| Before PR/release | `sync` (check for drift) |
| Project kickoff | `idea`, `roadmap`, `contributing`, `security` |
| Major technical decision | `adr` |
| Recurring user problem | `troubleshooting` |
| Roadmap change | `roadmap` |
