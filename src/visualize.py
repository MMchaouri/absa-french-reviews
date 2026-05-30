"""
Generate plots from results.json produced by:
    python runproject.py --save_results=True

Usage:
    python visualize.py                          # reads ../results.json
    python visualize.py --results_path=myfile.json
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.metrics import confusion_matrix

ASPECTS = ['Prix', 'Cuisine', 'Service']
LABELS = ['Positive', 'Négative', 'Neutre', 'NE']
COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']


def load_results(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def per_aspect_accuracy(preds, gts):
    accs = {}
    for aspect in ASPECTS:
        correct = sum(p[aspect] == g[aspect] for p, g in zip(preds, gts))
        accs[aspect] = round(100 * correct / len(gts), 1)
    return accs


def plot_comparison(results_list: list[dict], out_path: str):
    """Bar chart comparing accuracy per aspect across methods."""
    fig, ax = plt.subplots(figsize=(8, 5))

    n_methods = len(results_list)
    x = np.arange(len(ASPECTS))
    bar_w = 0.8 / n_methods

    for i, res in enumerate(results_list):
        accs = per_aspect_accuracy(res['predictions'], res['ground_truth'])
        label = res['method']
        if res['method'] == 'LLM':
            label += ' (3-shot)' if res.get('few_shot') else ' (0-shot)'
        vals = [accs[a] for a in ASPECTS]
        offset = (i - n_methods / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w, label=label, color=COLORS[i % len(COLORS)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(ASPECTS, fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title('Per-Aspect Accuracy by Method', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d%%'))
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def plot_confusion_matrices(res: dict, out_path: str):
    """One confusion matrix per aspect for a single result file."""
    preds, gts = res['predictions'], res['ground_truth']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    method_label = res['method']
    if res['method'] == 'LLM':
        method_label += ' (3-shot)' if res.get('few_shot') else ' (0-shot)'

    for ax, aspect in zip(axes, ASPECTS):
        y_true = [g[aspect] for g in gts]
        y_pred = [p[aspect] for p in preds]
        cm = confusion_matrix(y_true, y_pred, labels=LABELS)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=35, ha='right', fontsize=8)
        ax.set_yticklabels(LABELS, fontsize=8)
        ax.set_title(aspect, fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('True', fontsize=9)

        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                val = cm[i, j]
                color = 'white' if cm_norm[i, j] > 0.6 else 'black'
                ax.text(j, i, str(val), ha='center', va='center', fontsize=8, color=color)

    fig.suptitle(f'Confusion Matrices — {method_label}', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_path', default='../results.json')
    parser.add_argument('--compare', nargs='*', default=[],
                        help='Additional results.json files for the comparison chart')
    args = parser.parse_args()

    primary = load_results(args.results_path)
    out_dir = Path(args.results_path).parent

    plot_confusion_matrices(primary, str(out_dir / 'confusion_matrices.png'))

    all_results = [primary] + [load_results(p) for p in args.compare]
    if len(all_results) > 1:
        plot_comparison(all_results, str(out_dir / 'comparison_chart.png'))
    else:
        plot_comparison(all_results, str(out_dir / 'accuracy_chart.png'))


if __name__ == '__main__':
    main()
