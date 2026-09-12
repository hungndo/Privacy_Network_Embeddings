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

For a SLURM cluster, activate the intended Python environment or set `CN_VENV`,
then submit one of the four job arrays:

```bash
CN_ACCOUNT=my_account CN_PARTITION=gpu CN_VENV=/path/to/venv \
  ./submit_slurm_array.sh dense
```

The experiment names are `dense`, `sparse`, `dcsbm`, and `polblogs`. Each
submission writes to a new timestamped directory under `results/runs/`; it
never appends to the committed results. Scheduler defaults and their `CN_*`
overrides are documented by `./submit_slurm_array.sh` with no arguments. After
the array succeeds, run the merge command printed by the launcher to check for
missing shards, row-count errors, and duplicate experiment keys.

Use `./submit_slurm_array.sh --dry-run dense` to inspect the resulting `sbatch`
command without submitting or creating output. CPU-only runs require both
`CN_GRES=none` and `CN_REQUIRE_CUDA=0`; GPU availability is otherwise checked
on every worker before the experiment starts.

The committed result CSVs and figures predate CHANGE 6. They must not be
reported with the corrected privacy guarantee; regenerate them before updating
the paper.
