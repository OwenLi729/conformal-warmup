# conformal-warmup
Research Initiation Plan for Zavlanos Lab, Duke University

## Part 6 score extension

Run the three-seed cross-entropy/APS comparison from the repository root:

```bash
PYTHONPATH=src python src/acp/acp.py --run-score-extension
```

In Colab, paste `src/acp/acp_colab.py` and run:

```python
extension_results = run_score_extension_colab()
```

Both commands write summary JSON, raw adaptive-alpha values, two plots, and a
Markdown analysis to `score_extension_outputs/`.
