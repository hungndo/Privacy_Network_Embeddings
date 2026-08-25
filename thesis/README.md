# Thesis

Experiments, analysis, and figures for the thesis. Everything here uses paths
relative to this directory, so run scripts and notebooks with `thesis/` as the
working directory:

```bash
cd thesis
python script_sparse_synthetic_graph.py
jupyter lab analysis_sparse_synthetic_graph.ipynb
```

## Layout

```
thesis/
  script_*.py       experiment runners (write result CSVs)
  analysis_*.ipynb  read result CSVs, produce the figures
  plot_style.py     shared matplotlib styling, imported by the notebooks
  polblogs-1.gml    polblogs graph used by the polblogs experiment
  results/          result CSVs
  figures/          exported figures (.png / .pdf)
```

## Pipeline

| Experiment | Script | Results | Analysis | Figures |
| --- | --- | --- | --- | --- |
| Sparse synthetic graph | `script_sparse_synthetic_graph.py` | `results/results_dp_sbm_sparse_n500.csv` | `analysis_sparse_synthetic_graph.ipynb` | `figures/slide_nmi_sparse.png` |
| DCSBM synthetic graph | `script_dcsbm_synthetic_graph.py` | `results/results_dp_sbm_dcsbm_n500.csv` | `analysis_dcsbm_synthetic_graph.ipynb` | `figures/nmi_dcsbm_n500.png` |
| Polblogs | `script_polblogs.py` | `results/results_dp_sbm_polblogs.csv` | `analysis_polblogs.ipynb` | `figures/nmi_polblogs.png` |
| Unknown node labels | `script_no_known_node_label_synthetic_graph.py` | `results/results_dp_sbm_comparison_with_epochs.csv` | `analysis_no_known_node_label_synthetic_graph.ipynb` | `figures/slide_nmi_unknown_labels.png`, `figures/slide_beta_unknown_labels.png`, `figures/slide_epochs_unknown_labels.png` |
| Known node labels | `script_known_node_label_synthetic_graph.py` | `results/results.csv` | `analysis_known_node_label_synthetic_graph.ipynb` | — |
| EdgeFlip pure vs. approx DP | `script_edgeflip_pure_vs_approx_dp.py` | `results/results_edgeflip_pure_vs_approx_dp.csv` | `analysis_edgeflip_pure_vs_approx_dp.ipynb` | `figures/appendix_edgeflip_puredp_vs_approxdp.{png,pdf}`, `figures/appendix_edgeflip_delta_mechanics.{png,pdf}` |

Note: only `script_edgeflip_pure_vs_approx_dp.py` writes directly into
`results/`; the other scripts write their CSV to the current directory
(`OUTPUT_CSV` near the bottom of each script), so move the file into `results/`
before running the matching notebook. Sharded runs append `.shard<id>` to the
output filename.

Other files in `results/`: `results2.csv` and
`results_dp_sbm_sparse_n500_OLD.csv` are older runs kept for reference and are
not read by any notebook.
