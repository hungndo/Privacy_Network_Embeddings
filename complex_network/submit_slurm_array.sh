#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./submit_slurm_array.sh [--dry-run] EXPERIMENT [NUM_SHARDS]

EXPERIMENT is one of: dense, sparse, dcsbm, polblogs
Default task counts: dense=60, sparse=40, dcsbm=20, polblogs=100

Optional environment settings:
  CN_ACCOUNT           SLURM account (no default)
  CN_PARTITION         SLURM partition (no default)
  CN_TIME              wall time per task (default: 24:00:00)
  CN_MEM               memory per task (default: 32G)
  CN_CPUS               CPUs per task (default: 4)
  CN_GRES               generic resource (default: gpu:1; set "none" to omit)
  CN_MAX_PARALLEL       maximum simultaneous tasks (default: 10)
  CN_VENV               virtualenv directory to activate on each node
  CN_SETUP_SCRIPT       shell setup file to source on each node
  CN_PYTHON             Python executable after setup (default: python)
  CN_REQUIRE_CUDA       require torch CUDA support (default: 1)
  CN_RUN_NAME           output directory name (default: experiment_timestamp)

Example:
  CN_ACCOUNT=my_account CN_PARTITION=gpu CN_VENV=/path/to/venv \
    ./submit_slurm_array.sh dense
EOF
}

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=1
    shift
fi

experiment="${1:-}"
requested_shards="${2:-}"
case "$experiment" in
    dense)    default_shards=60; output_name="results_dp_sbm_complex_network.csv" ;;
    sparse)   default_shards=40; output_name="results_dp_sbm_sparse_n500.csv" ;;
    dcsbm)    default_shards=20; output_name="results_dp_sbm_dcsbm_n500.csv" ;;
    polblogs) default_shards=100; output_name="results_dp_sbm_polblogs.csv" ;;
    *) usage >&2; exit 2 ;;
esac

num_shards="${requested_shards:-$default_shards}"
if ! [[ "$num_shards" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_SHARDS must be a positive integer." >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_name="${CN_RUN_NAME:-${experiment}_${timestamp}}"
if ! [[ "$run_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "CN_RUN_NAME may contain only letters, digits, dots, underscores, and hyphens." >&2
    exit 2
fi
run_dir="$script_dir/results/runs/$run_name"
log_dir="$run_dir/logs"
output_csv="$run_dir/$output_name"

if [[ -e "$run_dir" ]]; then
    echo "Run directory already exists: $run_dir" >&2
    echo "Choose another CN_RUN_NAME to avoid mixing results." >&2
    exit 3
fi

sbatch_args=(
    --parsable
    --job-name="cn_${experiment}"
    --array="0-$((num_shards - 1))%${CN_MAX_PARALLEL:-10}"
    --time="${CN_TIME:-24:00:00}"
    --mem="${CN_MEM:-32G}"
    --cpus-per-task="${CN_CPUS:-4}"
    --output="$log_dir/%x-%A_%a.out"
    --chdir="$repo_root"
    --export="ALL,CN_EXPERIMENT=$experiment,CN_NUM_SHARDS=$num_shards,CN_REPO_ROOT=$repo_root,OUTPUT_CSV=$output_csv"
)

if [[ -n "${CN_ACCOUNT:-}" ]]; then
    sbatch_args+=(--account="$CN_ACCOUNT")
fi
if [[ -n "${CN_PARTITION:-}" ]]; then
    sbatch_args+=(--partition="$CN_PARTITION")
fi
if [[ "${CN_GRES:-gpu:1}" != "none" ]]; then
    sbatch_args+=(--gres="${CN_GRES:-gpu:1}")
fi

if [[ "$dry_run" == "1" ]]; then
    printf 'sbatch'
    printf ' %q' "${sbatch_args[@]}" "$script_dir/slurm_array_job.sh"
    printf '\n'
    echo "Dry run only; no job submitted or directory created."
    exit 0
fi

mkdir -p "$log_dir"
cat > "$run_dir/run.txt" <<EOF
experiment=$experiment
num_shards=$num_shards
output_csv=$output_csv
git_commit=$(git -C "$repo_root" rev-parse HEAD)
submitted_utc=$timestamp
EOF

job_id="$(sbatch "${sbatch_args[@]}" "$script_dir/slurm_array_job.sh")"
echo "Submitted $experiment as SLURM array job $job_id ($num_shards tasks)."
echo "Results: $run_dir"
echo "After completion, merge and validate with:"
echo "  python $script_dir/merge_slurm_results.py $experiment $run_dir --num-shards $num_shards"
