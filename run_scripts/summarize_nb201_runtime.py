#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import statistics
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize NB201 runtime profiling CSVs.')
    parser.add_argument('--raw-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    return parser.parse_args()


def to_float(v):
    try:
        return float(str(v).strip())
    except Exception:
        return None


def mean_std(values):
    if not values:
        return 0.0, 0.0
    m = statistics.fmean(values)
    if len(values) == 1:
        return m, 0.0
    return m, statistics.stdev(values)


def summarize_runs(paths):
    runs = []
    for path in paths:
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
        epoch_rows = [r for r in rows if r.get('row_type') == 'epoch_summary']
        step_rows = [r for r in rows if r.get('row_type') == 'step']
        if not epoch_rows:
            continue

        backend = epoch_rows[0].get('backend', '')
        batch_size = int(float(epoch_rows[0].get('batch_size', 0) or 0))

        epoch_times = [to_float(r.get('epoch_total_s')) for r in epoch_rows]
        epoch_times = [x for x in epoch_times if x is not None]
        mean_step_vals = [to_float(r.get('mean_step_s')) for r in epoch_rows]
        mean_step_vals = [x for x in mean_step_vals if x is not None]
        epoch_tput_vals = [to_float(r.get('samples_per_sec')) for r in epoch_rows]
        epoch_tput_vals = [x for x in epoch_tput_vals if x is not None]

        total_training_s = sum(epoch_times)
        mean_epoch_s = statistics.fmean(epoch_times) if epoch_times else 0.0
        mean_step_s = statistics.fmean(mean_step_vals) if mean_step_vals else 0.0
        mean_samples_per_sec = statistics.fmean(epoch_tput_vals) if epoch_tput_vals else 0.0
        step_total_vals = [to_float(r.get('step_total_s')) for r in step_rows]
        step_total_vals = [x for x in step_total_vals if x is not None and x > 0]
        mean_steps_per_sec = statistics.fmean([1.0 / x for x in step_total_vals]) if step_total_vals else 0.0

        runs.append({
            'backend': backend,
            'batch_size': batch_size,
            'total_training_s': total_training_s,
            'mean_epoch_s': mean_epoch_s,
            'mean_step_s': mean_step_s,
            'mean_samples_per_sec': mean_samples_per_sec,
            'mean_steps_per_sec': mean_steps_per_sec,
        })
    return runs


def write_outputs(out_dir, runs):
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, 'summary.csv')
    snippet_path = os.path.join(out_dir, 'paper_runtime_snippet.txt')

    grouped = defaultdict(list)
    for r in runs:
        grouped[(r['backend'], r['batch_size'])].append(r)

    rows = []
    for (backend, batch_size), group in sorted(grouped.items()):
        total_m, total_s = mean_std([x['total_training_s'] for x in group])
        epoch_m, epoch_s = mean_std([x['mean_epoch_s'] for x in group])
        step_m, step_s = mean_std([x['mean_step_s'] for x in group])
        stput_m, stput_s = mean_std([x['mean_samples_per_sec'] for x in group])
        ttput_m, ttput_s = mean_std([x['mean_steps_per_sec'] for x in group])
        rows.append({
            'backend': backend,
            'batch_size': batch_size,
            'n_runs': len(group),
            'total_training_s_mean': total_m,
            'total_training_s_std': total_s,
            'mean_epoch_s_mean': epoch_m,
            'mean_epoch_s_std': epoch_s,
            'mean_step_s_mean': step_m,
            'mean_step_s_std': step_s,
            'mean_samples_per_sec_mean': stput_m,
            'mean_samples_per_sec_std': stput_s,
            'mean_steps_per_sec_mean': ttput_m,
            'mean_steps_per_sec_std': ttput_s,
        })

    with open(summary_path, 'w', newline='') as f:
        fieldnames = list(rows[0].keys()) if rows else [
            'backend', 'batch_size', 'n_runs', 'total_training_s_mean', 'total_training_s_std',
            'mean_epoch_s_mean', 'mean_epoch_s_std', 'mean_step_s_mean', 'mean_step_s_std',
            'mean_samples_per_sec_mean', 'mean_samples_per_sec_std',
            'mean_steps_per_sec_mean', 'mean_steps_per_sec_std'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    with open(snippet_path, 'w') as f:
        f.write('NB201 runtime summary (fixed configuration)\n')
        f.write('----------------------------------------\n')
        for r in rows:
            f.write(
                f"Backend={r['backend']}, batch_size={r['batch_size']}, runs={r['n_runs']}: "
                f"total={r['total_training_s_mean']:.2f}±{r['total_training_s_std']:.2f}s, "
                f"epoch={r['mean_epoch_s_mean']:.2f}±{r['mean_epoch_s_std']:.2f}s, "
                f"step={r['mean_step_s_mean']:.4f}±{r['mean_step_s_std']:.4f}s, "
                f"throughput={r['mean_samples_per_sec_mean']:.2f}±{r['mean_samples_per_sec_std']:.2f} samples/s\n"
            )

    return summary_path, snippet_path


def main():
    args = parse_args()
    paths = sorted(glob.glob(os.path.join(args.raw_dir, '*.csv')))
    if not paths:
        raise SystemExit(f'No CSV files found in {args.raw_dir}')
    runs = summarize_runs(paths)
    if not runs:
        raise SystemExit('No valid profiling rows found.')
    summary_path, snippet_path = write_outputs(args.out_dir, runs)
    print(f'Saved: {summary_path}')
    print(f'Saved: {snippet_path}')


if __name__ == '__main__':
    main()
