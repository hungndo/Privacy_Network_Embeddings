#!/usr/bin/env bash
set -euo pipefail

# Worker invoked by submit_slurm_array.sh. Scheduler resources are supplied
# at submission time so account/partition conventions remain cluster-specific.

: "${CN_EXPERIMENT:?CN_EXPERIMENT is required}"
: "${CN_NUM_SHARDS:?CN_NUM_SHARDS is required}"
: "${CN_REPO_ROOT:?CN_REPO_ROOT is required}"
: "${OUTPUT_CSV:?OUTPUT_CSV is required}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as a SLURM array task}"

case "$CN_EXPERIMENT" in
    dense)    driver="script_no_known_node_label_synthetic_graph.py" ;;
    sparse)   driver="script_sparse_synthetic_graph.py" ;;
    dcsbm)    driver="script_dcsbm_synthetic_graph.py" ;;
    polblogs) driver="script_polblogs.py" ;;
    *) echo "Unknown experiment: $CN_EXPERIMENT" >&2; exit 2 ;;
esac

if [[ -n "${CN_SETUP_SCRIPT:-}" ]]; then
    # shellcheck source=/dev/null
    source "$CN_SETUP_SCRIPT"
fi
if [[ -n "${CN_VENV:-}" ]]; then
    # shellcheck source=/dev/null
    source "$CN_VENV/bin/activate"
fi

python_bin="${CN_PYTHON:-python}"
shard_path="${OUTPUT_CSV}.shard${SLURM_ARRAY_TASK_ID}"
if [[ -e "$shard_path" ]]; then
    echo "Refusing to append to existing shard: $shard_path" >&2
    echo "Remove that shard explicitly before resubmitting this array index." >&2
    exit 3
fi

cd "$CN_REPO_ROOT"

if [[ "${CN_REQUIRE_CUDA:-1}" == "1" ]]; then
    "$python_bin" -c \
        'import sys, torch; sys.exit(0 if torch.cuda.is_available() else "CUDA is not available")'
fi

echo "experiment=$CN_EXPERIMENT shard=$SLURM_ARRAY_TASK_ID/$CN_NUM_SHARDS"
echo "host=$(hostname) python=$($python_bin --version 2>&1) output=$shard_path"

"$python_bin" "complex_network/$driver" \
    --shard_id "$SLURM_ARRAY_TASK_ID" \
    --num_shards "$CN_NUM_SHARDS"
