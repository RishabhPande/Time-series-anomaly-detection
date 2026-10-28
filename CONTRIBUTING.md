# Contributing to Time-series-anomaly-detection

Thanks for helping improve this repo. Start with the README, which documents
how detection works, the quickstart, and the test coverage.

## Running the checks

```bash
python3 -m pytest tests/ -q
```

You can also run the end-to-end example with `python examples/quickstart.py`
(pure Python, no dependencies).

## Proposing a change

1. Fork the repo and create a branch for your change.
2. Make your change, add or update tests where it makes sense, and run the
   checks above.
3. Open a pull request describing what changed and why.

Note: every series in this repo is synthetic (see the README's Disclaimer),
so please keep new examples synthetic too.
