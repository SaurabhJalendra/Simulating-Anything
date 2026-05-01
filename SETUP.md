# Claude Code — First Session Setup

> Run this once on any new project: "Read SETUP.md and configure this project"
> Philosophy: Lean plugins, lean skills/agents, heavy CLAUDE.md rules.

---

## Step 1: Detect Project Stack

Scan the project root for:
- `package.json` → JavaScript/TypeScript
- `pyproject.toml` / `requirements.txt` / `setup.py` → Python
- `go.mod` → Go
- `Cargo.toml` → Rust
- `pom.xml` / `build.gradle` → Java/Kotlin
- `Gemfile` → Ruby
- `*.sln` / `*.csproj` → C#
- `Makefile` / `CMakeLists.txt` → C/C++

Also detect:
- Test framework (pytest, jest, go test, cargo test, etc.)
- Linter/formatter (prettier, black, eslint, rustfmt, etc.)
- Build command (npm run build, cargo build, go build, make, etc.)
- Package manager (npm, pip, cargo, go mod, etc.)

Report the detected stack before proceeding.

---

## Step 2: Install Core Plugins (Only 3)

```bash
claude plugin install superpowers
claude plugin install code-review
claude plugin install frontend-design    # Skip if no web frontend
```

**Why only 3?** Each plugin adds ~200 tokens to skill listings every turn. 3 plugins = ~600 tokens. The rest of the power comes from CLAUDE.md rules + 5 local skills + 3 local agents.

Verify: `claude plugin list`

---

## Step 3: Create `.claude/settings.json`

Ask the user: **personal or team project?**

**Personal project:**
```json
{
  "permissions": {
    "allow": [
      "Bash(*)", "Read(*)", "Write(*)", "Edit(*)",
      "Glob(*)", "Grep(*)", "WebSearch(*)", "WebFetch(*)",
      "Agent(*)", "Skill(*)", "NotebookEdit(*)",
      "mcp__claude-in-chrome__*"
    ]
  }
}
```

**Team project:**
```json
{
  "permissions": {
    "allow": [
      "Read(*)", "Glob(*)", "Grep(*)",
      "Bash(git status*)", "Bash(git diff*)", "Bash(git log*)"
    ],
    "deny": [
      "Bash(rm -rf*)", "Bash(git push --force*)", "Bash(git reset --hard*)"
    ]
  }
}
```

**If a formatter was detected (Step 1), add hooks:**
```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          { "type": "command", "command": "[detected formatter command] $CLAUDE_FILE_PATH 2>/dev/null || true" }
        ]
      }
    ]
  }
}
```

Examples: `npx prettier --write`, `black`, `rustfmt`, `gofmt -w`

---

## Step 4: Create Project CLAUDE.md

This is the **most important file** — it drives all behavior. Static content first, variable content last (cache optimization).

```markdown
# [Project Name]

## Stack
[Detected language, framework, database, key dependencies]

## Conventions
[Coding style, naming, file organization — ask user or infer from existing code]

## Architecture
[High-level system design — infer from codebase or ask user]

## Key Commands
- Build: [detected build command]
- Test: [detected test command]
- Lint: [detected lint command]
- Run: [detected run command]

---

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately
- Use plan mode for verification steps, not just building

### 2. Subagent Strategy
- Offload research, exploration, and parallel analysis to subagents
- One task per subagent for focused execution
- **Fork** (omit subagent_type) when the child needs conversation context — reuses KV cache
- **Explore** for quick codebase searches — uses Haiku, skips CLAUDE.md, fast and cheap
- **Plan** for architecture exploration — read-only, thorough
- **general-purpose** for independent implementation work
- Never delegate understanding — synthesize findings yourself before dispatching

### 3. Skill Activation
Use project skills instead of doing things manually:
- `/explore` — map the codebase before working in unfamiliar areas
- `/research` — research before building anything new
- `/commit` — stage, review, and commit with good messages
- `/debug` — systematic reproduce → isolate → fix → verify
- `/review` — review code for bugs, security, quality
- `/docs [mode]` — unified docs skill, 10 modes: changelog, pr, release, sync, idea, adr, roadmap, contributing, security, troubleshooting
- **Superpowers**: `brainstorming` → `writing-plans` → `executing-plans` → `verification-before-completion`

### 4. Parallel Execution
- Batch Read/Grep/Glob calls in ONE message — parallel, up to 10 concurrent
- Use `run_in_background: true` for long Bash commands
- Dispatch multiple Explore agents simultaneously for parallel research
- Keep Bash output under 30K chars — pipe through head/tail

### 5. Context Management
- Run `/compact` manually before context gets large
- Use `/compact Preserve findings about [topic]` for guided compaction
- Save important findings to memory for future sessions
- Old tool results (20+ turns) get snipped — summarize key info yourself
- For long output: say "complete implementation" upfront (8K → 64K escalation)

### 6. Verification Before Done
- Never mark a task complete without proving it works
- Run tests, check logs, demonstrate correctness
- Dispatch `verifier` agent for adversarial testing on important changes
- Ask: "Would a staff engineer approve this?"

### 7. Self-Improvement
- After ANY correction: save the pattern to `tasks/lessons.md`
- Review lessons at session start
- Write rules that prevent the same mistake

### 8. Autonomous Execution
- Do NOT ask for permission to read files, search the web, or run safe commands
- Do NOT ask "would you like me to..." — just do it
- Only pause for: destructive git operations, sending external messages, spending money

---

## Auto-Triggers (These Replace 15+ Skills/Agents — Zero Token Cost)

### After Editing Code
- Run tests if they exist — report pass/fail
- Check if documentation (README, API docs, docstrings) references changed code — update if stale
- If 3+ tasks completed without verification, dispatch `verifier` agent

### Before Committing
- Review changes for bugs, security issues, convention violations
- Update CHANGELOG.md if it exists (add entry for what changed)
- Use `/commit` skill for proper staging and commit message

### Before PR / Release
- Dispatch `reviewer` agent for full code review
- Check: do tests pass? are docs current? any security concerns?
- Generate PR description: summary, changes, testing, notes
- Generate release notes if creating a release

### At Session Start
- Check `git status` and recent commits
- Read `tasks/lessons.md` for past learnings
- Check if dependencies look outdated (suggest update if so)
- Report brief project status before starting work

### When Patterns Repeat
- If you do the same sequence 3+ times, create a skill for it
  - Write SKILL.md to `.claude/skills/<name>/SKILL.md`
  - Needs: precise 250-char description, allowed-tools, clear instructions
  - Tell the user what you created
- If you need a specialized agent for a recurring task, create one
  - Write to `.claude/agents/<name>.md`
  - Keep system prompt under 50 lines, minimum tools needed

### Documentation Auto-Maintenance
**A project with excellent docs is automatically a higher-class project. Docs matter — treat them as first-class.**

After ANY meaningful code change:
- Dispatch `documenter` agent to update all affected docs (README, CHANGELOG, API docs, architecture, docstrings)
- Run `/changelog` to add entry to CHANGELOG.md
- Run `/docs-sync` if uncertain whether docs are current

Triggers:
- New project with no README → generate one immediately
- Any code change → update CHANGELOG (use `/changelog`)
- API endpoints changed → update API docs
- Architecture changed → update architecture docs
- New functions/classes → add docstrings in same edit
- Setup process changed → update README setup section
- Breaking change → CHANGELOG `### Changed` + bump version + update migration notes
- Before PR → run `/docs-sync` to catch any drift
- Before release → run `/changelog release` for user-facing notes

**Rule: never ship code changes without corresponding doc updates in the same session.**

---

## Cache Optimization
- This file: static content first, variable content last
- Do NOT edit this file mid-session — busts the entire prompt cache
- Keep MCP server list stable within a session
- Fork subagents reuse parent's KV cache

---

## Memory System
- Save architecture decisions, user preferences, key findings, feedback
- Descriptive file names and one-line summaries in MEMORY.md index
- Don't save: code patterns from codebase, git history, ephemeral details

---

## Current Focus (update BETWEEN sessions only)
[Ask user what they're working on — LAST because it changes]
```

Fill in what you can detect automatically. Ask the user for what you can't.

---

## Step 5: Build Local Skills (6)

### 5a. Explore Codebase Skill

`.claude/skills/explore/SKILL.md`:
```yaml
---
name: explore-codebase
description: Deep codebase exploration — maps architecture, finds patterns, documents structure
allowed-tools: [Read, Grep, Glob, Agent]
---

# Explore Codebase

## Steps
1. Map the directory structure (Glob for key file patterns)
2. Identify the entry point(s)
3. Trace the main data flow / request lifecycle
4. Identify key abstractions and patterns
5. Document findings in memory

## Output
- Directory tree with annotations
- Key files and their purposes
- Architecture patterns identified
- Data flow diagram (text)
- Save findings to memory for future sessions
```

### 5b. Research Before Build Skill

`.claude/skills/research/SKILL.md`:
```yaml
---
name: research
description: Research a topic thoroughly before implementation — web search, papers, docs, existing code
allowed-tools: [WebSearch, WebFetch, Read, Grep, Glob, Agent]
---

# Research Before Building

## Steps
1. Search the web for current best practices on $ARGUMENTS
2. Check Context7 for relevant library docs (use context7)
3. Search the existing codebase for related patterns
4. If academic: use Consensus for papers
5. Synthesize findings into actionable recommendations

## Output
- Summary of findings with sources
- Recommended approach with trade-offs
- Relevant code examples from codebase
- Save key findings to memory
```

### 5c. Commit Skill

`.claude/skills/commit/SKILL.md`:
```yaml
---
name: commit
description: Creates a well-formatted git commit — reviews changes, writes message, commits
allowed-tools: [Bash, Read, Grep]
---

# Smart Commit

## Steps
1. Run `git status` and `git diff --cached` (or `git diff` if nothing staged)
2. Analyze what changed and why
3. Stage relevant files (specific files, not `git add .`)
4. Write commit message:
   - First line: type + imperative description, under 72 chars
   - Types: feat, fix, refactor, docs, test, chore, perf
   - Body (if needed): what changed and why
5. Commit with the message
6. Show the commit hash

## Rules
- Never commit .env, credentials, or secrets
- Never use --no-verify
- Prefer specific file staging over `git add .`
- If nothing changed, say so
```

### 5d. Debug Skill

`.claude/skills/debug/SKILL.md`:
```yaml
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
```

### 5e. Code Review Skill

`.claude/skills/review/SKILL.md`:
```yaml
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
   - **Conventions**: Does it follow project patterns?
   - **Tests**: Are changes tested?
3. Check types: run typecheck if available
4. Check lint: run linter if available

## Output Format
For each issue:
- CRITICAL / WARNING / SUGGESTION
- File:line — what's wrong — how to fix

If no issues: "Code looks good. No issues found."
```

### 5f. Docs Skill (Unified Documentation Manager)

`.claude/skills/docs/SKILL.md`:
```yaml
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
## Non-Goals   ← Critical: prevents scope creep
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
## Not Doing   ← Critical: prevents same rejected requests coming back
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
## Reporting a Vulnerability   ← Private channel, not GitHub issue
## Supported Versions
## Security Considerations   ← Data handling, auth, deps, secrets
## Known Limitations
## Security History   ← CVEs fixed
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
```

## Step 6: Build Local Agents (4)

### 6a. Reviewer Agent

`.claude/agents/reviewer.md`:
```yaml
---
name: reviewer
description: Reviews code changes against project conventions, security, and quality standards
tools: [Read, Grep, Glob, Bash]
model: opus
color: yellow
---

# Code Reviewer

You review code for the current project. Be critical — don't rubber-stamp.

## Check
1. `git diff` to see changes
2. Read changed files in full
3. Check for: bugs, security issues, convention violations, missing tests
4. Check types and lint if commands exist

## Report
- CRITICAL: Must fix (blocks merge)
- WARNING: Should fix
- SUGGESTION: Nice to have
- For each: file:line, what's wrong, how to fix

Be specific. Be harsh. Better to catch it now.
```

### 6b. Verifier Agent

`.claude/agents/verifier.md`:
```yaml
---
name: verifier
description: Adversarially tests implementations — tries to break things, runs edge cases
tools: [Read, Bash, Grep, Glob]
model: opus
color: red
---

# Adversarial Verifier

Your job is to TRY TO BREAK the implementation. Not confirm it works.

## Approach
1. Read what was implemented
2. Run the tests — do they actually test the right things?
3. Try edge cases the tests don't cover
4. Try error paths — what happens with bad input?
5. Check: does the fix actually fix the ROOT CAUSE?

## Rules
- Don't trust the implementation — verify independently
- Don't trust the tests — they were written by the same agent
- If you can't break it after genuine effort, say "verified"
- If you CAN break it, report exactly how with reproduction steps
```

### 6c. Researcher Agent

`.claude/agents/researcher.md`:
```yaml
---
name: researcher
description: Deep research on a topic using web search, papers, docs, and codebase analysis
tools: [WebSearch, WebFetch, Read, Grep, Glob]
model: opus
color: blue
omitClaudeMd: true
---

# Research Agent

Research the given topic thoroughly.

## Steps
1. Web search for current best practices and approaches
2. Check official documentation (use Context7 if library-related)
3. Search the codebase for existing related patterns
4. Synthesize findings

## Output
- Key findings (bullet points, with source links)
- Recommended approach
- Trade-offs and alternatives
- Relevant existing code in this project
```

### 6d. Documenter Agent

`.claude/agents/documenter.md`:
```yaml
---
name: documenter
description: Maintains all project documentation — README, CHANGELOG, API docs, architecture docs, docstrings. Keeps docs in sync with code.
tools: [Read, Write, Edit, Bash, Grep, Glob]
model: opus
color: white
---

# Documentation Maintainer

You maintain all project documentation. A project with excellent docs is automatically a higher-class project.

## What You Maintain

| Document | When to Update |
|---|---|
| `README.md` | Whenever public API, setup steps, or core features change |
| `CHANGELOG.md` | After every meaningful change (feature, fix, breaking change) |
| `docs/api.md` | When API endpoints, parameters, or responses change |
| `docs/architecture.md` | When system design, data flow, or key patterns change |
| `docs/onboarding.md` | When setup or dev workflow changes |
| Inline docstrings | When functions/classes are added or modified |

## Workflow

### When Dispatched After Code Changes
1. Run `git diff HEAD~1` to see what changed
2. For each changed file:
   - Does it affect the README? (public API, setup, features)
   - Does it need CHANGELOG entry? (yes, almost always)
   - Does it change API surface? (update `docs/api.md`)
   - Does it change architecture? (update `docs/architecture.md`)
   - Does it have undocumented functions/classes? (add docstrings)
3. Update every affected doc
4. Add to CHANGELOG.md under `## [Unreleased]`:
   - `### Added` for new features
   - `### Changed` for modifications
   - `### Fixed` for bug fixes
   - `### Removed` for deletions
5. Report what was updated

### When Dispatched for Full Audit
1. Read all current docs
2. Check each against current codebase:
   - Is the README's "features" list accurate?
   - Are setup steps current?
   - Are API docs matching actual endpoints?
   - Is architecture doc matching actual structure?
3. Report every discrepancy with line references
4. Fix or flag for user decision

## Rules
- Match existing doc style exactly — if docs use emojis, use them; if not, don't
- Never duplicate information across docs — link to the source of truth
- Focus on WHY, not WHAT (code shows WHAT)
- Include realistic code examples, not toy ones
- Update, don't rewrite — preserve history and context
- NEVER leave broken links or references to deleted features
```

---

## Step 7: Create Output Style

`.claude/output-styles/engineering.md`:
```yaml
---
name: engineering
description: Concise engineering output — code first, explanation second
---

- Lead with code or commands, not explanation
- Use tables for comparisons
- Show the fix, not the theory
- Keep responses focused — don't repeat the question back
- If showing a file change, show only the changed parts
```

---

## Step 8: Initialize Memory

Create project memory:

**MEMORY.md** (index):
```markdown
# [Project Name] Memory Index

- [Project Purpose](project_purpose.md) — What this project does and why
- [User Profile](user_profile.md) — User's role, preferences, how they work
```

**project_purpose.md:**
```yaml
---
name: Project Purpose
description: What [project name] does, its goals, and key constraints
type: project
---

[Ask user to describe the project. Save their response here.]
```

**user_profile.md** — skip if global profile already exists.

---

## Step 9: Optional — LLM Wiki (Karpathy Pattern)

**Only do this if the project involves accumulating knowledge from external sources** — research, learning, reading many papers/articles, competitive analysis, book notes, etc.

**Ask the user:** "Does this project involve reading and synthesizing many external sources (papers, articles, docs)? If yes, set up the LLM Wiki layer."

If yes, create this structure:

```
raw/                    ← Immutable source documents you drop in (papers, articles, clippings)
└── assets/            ← Images, PDFs, referenced files

wiki/                   ← LLM-maintained markdown knowledge base
├── index.md           ← Content catalog (all pages, by category, with one-line summaries)
├── log.md             ← Append-only activity record with timestamps
├── entities/          ← Pages for people, organizations, products, concepts
├── concepts/          ← Pages for ideas, theories, methods
├── sources/           ← One page per ingested source (summary + key takeaways)
└── syntheses/         ← Cross-cutting analyses, comparisons, theses
```

Initialize `wiki/index.md`:
```markdown
# Wiki Index

## Entities
(no entities yet)

## Concepts
(no concepts yet)

## Sources
(no sources yet)

## Syntheses
(no syntheses yet)
```

Initialize `wiki/log.md`:
```markdown
# Activity Log

## [YYYY-MM-DD] setup | Wiki initialized
Empty wiki structure created. Ready to ingest sources.
```

Add this section to the project's CLAUDE.md (after "Auto-Triggers"):

```markdown
---

## LLM Wiki Operations

This project uses the Karpathy LLM Wiki pattern. `raw/` holds immutable sources, `wiki/` holds LLM-maintained knowledge.

### Ingest (when user drops a file in raw/)
1. Read the source thoroughly
2. Write a summary page in `wiki/sources/[source-name].md`
3. Update 10-15 related pages in `wiki/entities/` and `wiki/concepts/`
4. Update `wiki/index.md` with new entries
5. Append entry to `wiki/log.md`: `## [YYYY-MM-DD] ingest | [source name]`
6. Report to user: what was ingested, which pages were updated

### Query (when user asks a question)
1. Read `wiki/index.md` first to find relevant pages
2. Drill into those pages
3. Synthesize an answer with citations to source pages
4. **If the answer is valuable**, file it back as a new page in `wiki/syntheses/`
5. Append to `wiki/log.md`: `## [YYYY-MM-DD] query | [topic]`

### Lint (periodic health check — user runs /wiki-lint)
Check for:
- Contradictions between pages (flag for user decision)
- Stale claims that newer sources have superseded
- Orphan pages with no inbound links
- Important concepts mentioned but lacking their own page
- Missing cross-references between related pages
- Data gaps that could be filled with web search

### Rules
- Never modify files in `raw/` — those are immutable
- Every wiki page should link to its sources
- Use YAML frontmatter on wiki pages: name, description, type, sources, last_updated
- Keep `index.md` under 200 lines — one line per page
```

Also create an optional `wiki-lint` skill:

`.claude/skills/wiki-lint/SKILL.md`:
```yaml
---
name: wiki-lint
description: Health-check the wiki for contradictions, orphan pages, stale claims, missing cross-references
allowed-tools: [Read, Grep, Glob, WebSearch]
---

# Wiki Lint

Run a health check on the wiki.

## Steps
1. Read `wiki/index.md`
2. For each page, check:
   - Does it have inbound links? (grep for its name across wiki/)
   - Does it cite sources?
   - Are referenced pages actually in the index?
3. Cross-check claims:
   - Find contradictions between pages
   - Find claims without source citations
4. Identify gaps:
   - Concepts mentioned but lacking their own page
   - Entities referenced but not documented
5. Report findings with specific page references and suggested fixes
```

---

## Step 9.5: Optional — GitHub Actions Integration

**Only do this if the project is on GitHub AND the user wants AI-powered PR reviews/issue fixing.**

**Ask the user:** "Do you want Claude Code integrated with GitHub Actions? This enables `@claude` mentions on PRs/issues for automated reviews and fixes. Requires a Claude API token added as a GitHub secret."

If yes:

### Setup Steps

1. **Install the GitHub App:**
   - Go to https://github.com/apps/claude
   - Install on the target repository

2. **Add API token as GitHub secret:**
   - Repo Settings → Secrets and variables → Actions → New repository secret
   - Name: `ANTHROPIC_API_KEY` (or `CLAUDE_CODE_OAUTH_TOKEN` for OAuth)
   - Value: User's Anthropic API key

3. **Create `.github/workflows/claude.yml`:**

```yaml
name: Claude Code
on:
  issue_comment:
    types: [created]
  pull_request_review_comment:
    types: [created]
  issues:
    types: [opened, assigned]

jobs:
  claude:
    if: |
      (github.event_name == 'issue_comment' && contains(github.event.comment.body, '@claude')) ||
      (github.event_name == 'pull_request_review_comment' && contains(github.event.comment.body, '@claude')) ||
      (github.event_name == 'issues' && contains(github.event.issue.body, '@claude'))
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
      issues: write
    steps:
      - uses: actions/checkout@v4
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
```

### Usage After Setup

| Action | How |
|---|---|
| **Auto-review a PR** | Add comment `@claude review this PR` |
| **Fix from issue** | Create issue with `@claude fix this` |
| **Ask about code** | Comment `@claude how does this function work?` |
| **Implement feature** | Issue body: `@claude implement [description]` |

### Rules
- Never commit API keys directly — always use GitHub secrets
- Set spending limits on your Anthropic account to avoid runaway costs
- Consider restricting the workflow to specific repo maintainers via `github.event.comment.user.login`
- For private repos, OAuth token (`CLAUDE_CODE_OAUTH_TOKEN`) works better than API key

### Advanced: Custom CLAUDE.md for GitHub Actions

Add `.github/claude.md` to override project CLAUDE.md for GitHub Actions context only (different rules for PR review vs local development).

---

## Step 10: Initialize Project Structure

Create any missing directories:
```
.claude/
├── settings.json            ← Created in Step 3
├── skills/                  ← Created in Step 5
│   ├── explore/SKILL.md
│   ├── research/SKILL.md
│   ├── commit/SKILL.md
│   ├── debug/SKILL.md
│   ├── review/SKILL.md
│   ├── docs/SKILL.md        ← Unified docs skill (10 modes)
│   └── wiki-lint/SKILL.md   ← Only if Step 9 ran
├── agents/                  ← Created in Step 6
│   ├── reviewer.md
│   ├── verifier.md
│   ├── researcher.md
│   └── documenter.md
└── output-styles/           ← Created in Step 7
    └── engineering.md
docs/
└── superpowers/
    └── specs/               ← For brainstorming design specs
tasks/
└── lessons.md               ← Self-improvement log (start empty)

raw/                         ← Only if Step 9 ran (knowledge project)
└── assets/
wiki/                        ← Only if Step 9 ran (knowledge project)
├── index.md
├── log.md
├── entities/
├── concepts/
├── sources/
└── syntheses/
```

---

## Step 11: Report Setup Summary

```
| Component          | Status      | Details                          |
|-------------------|-------------|-----------------------------------|
| Stack             | [language]  | [framework, test runner, linter]   |
| Plugins           | 3 installed | superpowers, code-review, frontend |
| .claude/settings  | created     | [permission mode] + [hooks]        |
| CLAUDE.md         | created     | [X lines] with workflow rules      |
| Skills            | 6 or 7      | explore, research, commit, debug, review, docs (10 modes) [+ wiki-lint if Step 9 ran] |
| Agents            | 4 created   | reviewer, verifier, researcher, documenter |
| LLM Wiki          | optional    | raw/ + wiki/ structure if Step 9 ran |
| GitHub Actions    | optional    | .github/workflows/claude.yml if Step 9.5 ran |
| Output Style      | created     | engineering                        |
| Memory            | initialized | [X memory files]                   |
```

---

## Step 12: Ask "What are we building?"

After setup is complete, ask the user what they want to work on. Their answer:
1. Gets saved to memory (project purpose)
2. Gets added to CLAUDE.md "Current Focus"
3. Kicks off the first task
