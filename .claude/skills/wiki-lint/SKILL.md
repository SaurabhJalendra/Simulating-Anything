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
