#!/usr/bin/env python3
"""Validate and merge the CSV shards produced by submit_slurm_array.sh."""

import argparse
import csv
from pathlib import Path


EXPERIMENTS = {
    "dense": ("results_dp_sbm_complex_network.csv", 900),
    "sparse": ("results_dp_sbm_sparse_n500.csv", 600),
    "dcsbm": ("results_dp_sbm_dcsbm_n500.csv", 300),
    "polblogs": ("results_dp_sbm_polblogs.csv", 300),
}


def shard_number(path, output_name):
    return int(path.name.removeprefix(f"{output_name}.shard"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    output_name, expected_rows = EXPERIMENTS[args.experiment]
    output_path = args.run_dir / output_name
    shards = sorted(
        args.run_dir.glob(f"{output_name}.shard*"),
        key=lambda path: shard_number(path, output_name),
    )
    expected_numbers = list(range(args.num_shards))
    actual_numbers = [shard_number(path, output_name) for path in shards]
    if actual_numbers != expected_numbers:
        missing = sorted(set(expected_numbers) - set(actual_numbers))
        unexpected = sorted(set(actual_numbers) - set(expected_numbers))
        raise SystemExit(f"Shard mismatch: missing={missing}, unexpected={unexpected}")
    if output_path.exists():
        raise SystemExit(f"Refusing to overwrite existing merged file: {output_path}")

    fieldnames = None
    rows = []
    for shard in shards:
        with shard.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            elif reader.fieldnames != fieldnames:
                raise SystemExit(f"CSV header differs in {shard}")
            rows.extend(reader)

    if len(rows) != expected_rows:
        raise SystemExit(f"Expected {expected_rows} rows, found {len(rows)}")

    key_fields = {
        "dense": ("N", "sigma", "rep", "method"),
        "sparse": ("regime", "N", "sigma", "rep", "method"),
        "dcsbm": ("N", "sigma", "rep", "method"),
        "polblogs": ("N", "sigma", "rep", "method"),
    }[args.experiment]
    keys = [tuple(row[field] for field in key_fields) for row in rows]
    if len(keys) != len(set(keys)):
        raise SystemExit(f"Duplicate experiment rows found using key {key_fields}")

    with output_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Merged and validated {len(rows)} rows into {output_path}")


if __name__ == "__main__":
    main()
