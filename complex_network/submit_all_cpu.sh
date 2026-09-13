#!/usr/bin/env bash
set -euo pipefail

# Submit the complete corrected experiment suite to CPU-only SLURM nodes.
# The access node only executes this lightweight submission section; package
# installation, tests, experiments, and merges all run as scheduled jobs.

EXPERIMENT_DIR="${CN_EXPERIMENT_DIR:-/home/oward/scratch/Privacy_Network_Embeddings/complex_network}"
REPO_ROOT="$(dirname "$EXPERIMENT_DIR")"
VENV_DIR="${CN_VENV:-$REPO_ROOT/.venv}"

# Adjust these through environment variables if the cluster uses other names.
COMPILER_MODULE="${CN_COMPILER_MODULE:-gcc}"
PYTHON_MODULE="${CN_PYTHON_MODULE:-python}"
PYTHON_COMMAND="${CN_PYTHON_COMMAND:-python3}"

CPUS="${CN_CPUS:-6}"
MEM_PER_CPU="${CN_MEM_PER_CPU:-8000}"
TIME_LIMIT="${CN_TIME:-2000:00}"
MAX_PARALLEL="${CN_MAX_PARALLEL:-4}"
MAIL_USER="${CN_MAIL_USER:-oward@sfu.ca}"

load_modules() {
    module load "$COMPILER_MODULE"
    module load "$PYTHON_MODULE"
}

submit_job() {
    if [[ "${CN_DRY_RUN:-0}" == "1" ]]; then
        printf 'sbatch' >&2
        printf ' %q' "$@" >&2
        printf '\n' >&2
        echo 999999
    else
        local reply
        reply="$(sbatch --parsable "$@")"
        echo "${reply%%;*}"
    fi
}

mode="${1:-submit}"

if [[ "$mode" == "setup" ]]; then
    cd "$EXPERIMENT_DIR"
    load_modules

    if [[ ! -x "$VENV_DIR/bin/python" ]]; then
        "$PYTHON_COMMAND" -m venv "$VENV_DIR"
    fi

    "$VENV_DIR/bin/python" -m pip install --upgrade pip
    "$VENV_DIR/bin/python" -m pip install \
        --index-url https://download.pytorch.org/whl/cpu torch
    "$VENV_DIR/bin/python" -m pip install -r requirements.txt
    "$VENV_DIR/bin/python" -m unittest -v test_privacy_fixes.py
    exit 0
fi

if [[ "$mode" == "merge" ]]; then
    : "${CN_EXPERIMENT:?CN_EXPERIMENT is required}"
    : "${CN_RUN_DIR:?CN_RUN_DIR is required}"
    : "${CN_NUM_SHARDS:?CN_NUM_SHARDS is required}"
    cd "$EXPERIMENT_DIR"
    load_modules
    "$VENV_DIR/bin/python" merge_slurm_results.py \
        "$CN_EXPERIMENT" "$CN_RUN_DIR" --num-shards "$CN_NUM_SHARDS"
    exit 0
fi

if [[ "$mode" == "worker" ]]; then
    cd "$EXPERIMENT_DIR"
    load_modules
    exec "$EXPERIMENT_DIR/slurm_array_job.sh"
fi

if [[ "$mode" != "submit" ]]; then
    echo "Usage: $0 [submit]" >&2
    exit 2
fi

if [[ ! -d "$EXPERIMENT_DIR" ]]; then
    echo "Experiment directory does not exist: $EXPERIMENT_DIR" >&2
    exit 2
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_root="$EXPERIMENT_DIR/results/runs/cpu_${timestamp}"
log_dir="$run_root/logs"
mkdir -p "$log_dir"

script_path="$EXPERIMENT_DIR/submit_all_cpu.sh"

common_job_args=(
    --cpus-per-task="$CPUS"
    --mem-per-cpu="$MEM_PER_CPU"
    --time="$TIME_LIMIT"
    --mail-user="$MAIL_USER"
    --mail-type=FAIL
)

setup_job="$(submit_job \
    --job-name=cn_setup \
    --cpus-per-task=1 \
    --mem=8G \
    --time=02:00:00 \
    --mail-user="$MAIL_USER" \
    --mail-type=FAIL \
    --output="$log_dir/setup_%j.out" \
    --export="ALL,CN_EXPERIMENT_DIR=$EXPERIMENT_DIR,CN_VENV=$VENV_DIR,CN_COMPILER_MODULE=$COMPILER_MODULE,CN_PYTHON_MODULE=$PYTHON_MODULE,CN_PYTHON_COMMAND=$PYTHON_COMMAND" \
    "$script_path" setup)"

printf 'setup=%s\n' "$setup_job" > "$run_root/job_ids.txt"
echo "Submitted environment setup/tests: $setup_job"

submit_experiment() {
    local experiment="$1"
    local num_shards="$2"
    local output_name="$3"
    local experiment_dir="$run_root/$experiment"
    mkdir -p "$experiment_dir"

    local array_job
    array_job="$(submit_job \
        --job-name="cn_${experiment}" \
        --dependency="afterok:$setup_job" \
        --array="0-$((num_shards - 1))%$MAX_PARALLEL" \
        "${common_job_args[@]}" \
        --output="$log_dir/${experiment}_%A_%a.out" \
        --export="ALL,CN_EXPERIMENT_DIR=$EXPERIMENT_DIR,CN_COMPILER_MODULE=$COMPILER_MODULE,CN_PYTHON_MODULE=$PYTHON_MODULE,CN_EXPERIMENT=$experiment,CN_NUM_SHARDS=$num_shards,CN_REPO_ROOT=$REPO_ROOT,CN_VENV=$VENV_DIR,CN_REQUIRE_CUDA=0,OUTPUT_CSV=$experiment_dir/$output_name" \
        "$script_path" worker)"

    local merge_job
    merge_job="$(submit_job \
        --job-name="cn_merge_${experiment}" \
        --dependency="afterok:$array_job" \
        --cpus-per-task=1 \
        --mem=2G \
        --time=00:10:00 \
        --mail-user="$MAIL_USER" \
        --mail-type=FAIL,END \
        --output="$log_dir/merge_${experiment}_%j.out" \
        --export="ALL,CN_EXPERIMENT_DIR=$EXPERIMENT_DIR,CN_VENV=$VENV_DIR,CN_COMPILER_MODULE=$COMPILER_MODULE,CN_PYTHON_MODULE=$PYTHON_MODULE,CN_EXPERIMENT=$experiment,CN_NUM_SHARDS=$num_shards,CN_RUN_DIR=$experiment_dir" \
        "$script_path" merge)"

    printf '%s_array=%s\n%s_merge=%s\n' \
        "$experiment" "$array_job" "$experiment" "$merge_job" \
        >> "$run_root/job_ids.txt"
    echo "Submitted $experiment array: $array_job; merge check: $merge_job"
}

submit_experiment dense 60 results_dp_sbm_complex_network.csv
submit_experiment sparse 40 results_dp_sbm_sparse_n500.csv
submit_experiment dcsbm 20 results_dp_sbm_dcsbm_n500.csv
submit_experiment polblogs 100 results_dp_sbm_polblogs.csv

echo
echo "Run directory: $run_root"
echo "Job IDs:       $run_root/job_ids.txt"
echo "Task logs:     $log_dir/<experiment>_<array-job>_<task>.out"
echo "Setup log:     $log_dir/setup_${setup_job}.out"
echo
echo "Monitor with:"
echo "  squeue -u $USER"
echo "  tail -f $log_dir/*.out"
echo "  sacct -j $setup_job --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode"
echo
echo "Find likely failures in the logs with:"
echo "  grep -iEn 'traceback|error|failed|out of memory|oom|time limit|killed' $log_dir/*.out"
