# Complex Networks experiments

The four experiment drivers share the same SBM-DPSGD implementation. Privacy
audit changes are marked `[CHANGE 6a]` through `[CHANGE 6g]` in each file:

- unique unordered dyads sampled without replacement;
- `dp-accounting` RDP composition under `REPLACE_ONE` adjacency;
- the data-independent `rest/M` gradient added after clipping and noise;
- a symmetric Beta variational posterior;
- noise multipliers recalibrated to approximately the original epsilon grid;
- sparse analysis fixed to read the n=500 CSV;
- sparse experiments retain the full fixed vertex set.

Install the dependencies and run the focused checks with:

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_privacy_fixes.py
```

Run an experiment from this directory, for example:

```bash
python script_no_known_node_label_synthetic_graph.py
```

The committed result CSVs and figures predate CHANGE 6. They must not be
reported with the corrected privacy guarantee; regenerate them before updating
the paper.
