# raw/

Drop source documents here — papers (PDFs), articles, clippings, book notes, transcripts. Anything you want the LLM to read once and remember.

## Rules
- **Files in this directory are immutable.** Claude reads them but does not edit them.
- For binary files (PDFs, images), use `assets/` subfolder.
- Filename convention: `[YYYY-MM-DD]-[slug].md` for clippings, original filename for papers.

## Workflow
1. Drop a file here.
2. Tell Claude: "Ingest the new file in raw/."
3. Claude reads it, writes `wiki/sources/[filename].md` with summary, and updates 10-15 related pages in `wiki/entities/` and `wiki/concepts/`.

See `wiki/index.md` and CLAUDE.md "LLM Wiki Operations" for full protocol.
