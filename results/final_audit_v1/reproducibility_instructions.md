# Reproducibility instructions

From the repository root, run:

```bash
python3 -m scripts.analysis.finalize_research_results --check-reproducibility
```

The command validates every existing branch hash manifest and regenerates this audit from frozen CSV/JSON/source artifacts. It does not parse or write `.osr`, recalibrate offsets, rematch the raw corpus, access the osu! API, or fit a model. It fails on any frozen manifest mismatch and compares a temporary clean regeneration byte-for-byte with the canonical generated artifacts (excluding the self-referential output manifest and the reproducibility report).
