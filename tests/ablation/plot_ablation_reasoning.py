"""
Generate plots from reasoning layer ablation results.

Creates summary plots for Flickr30k referring expression grounding.

Usage:
    python plot_ablation_reasoning.py --summary path/to/summary.json --output_dir path/to/save/
"""

import os
import json
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_results(summary_path):
    with open(summary_path) as f:
        results = json.load(f)
    results = sorted(results, key=lambda r: r['extract_layer'])
    return results


def make_summary_plot(results, output_path):
    """Static multi-metric comparison for reasoning."""
    layers = [r['extract_layer'] for r in results]
    accuracy = [r['accuracy'] for r in results]
    precision = [r['precision'] for r in results]
    recall = [r['recall'] for r in results]
    fp_rate = [r['fp_rate'] for r in results]
    
    n = len(layers)
    if n == 0: return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle('ThinkDet — Reasoning Ablation (Flickr30k Referring Expressions)',
                fontsize=15, color='#f0f6fc', fontweight='bold', y=0.98)

    configs = [
        (axes[0,0], accuracy,   'Accuracy',       '#58a6ff', True),
        (axes[0,1], precision,  'Precision',      '#3fb950', True),
        (axes[1,0], recall,     'Recall',         '#d29922', True),
        (axes[1,1], fp_rate,    'FP Rate',        '#f85149', False),
    ]

    for ax, vals, title, color, higher_better in configs:
        ax.set_facecolor('#0d1117')
        
        best_idx = int(np.argmax(vals)) if higher_better else int(np.argmin(vals))
        bar_colors = [color if i != best_idx else '#f0f6fc' for i in range(n)]
        
        ax.bar(range(n), vals, color=bar_colors, edgecolor='#30363d',
               linewidth=0.3, width=0.75, alpha=0.85)
        ax.plot(range(n), vals, color=color, linewidth=1.5, alpha=0.7,
                marker='o', markersize=3)
        
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(l) for l in layers], fontsize=6, color='#8b949e')
        ax.set_title(title, fontsize=11, color=color, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=8, color='#8b949e')
        ax.tick_params(colors='#8b949e', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#30363d')
        ax.grid(axis='y', alpha=0.1, color='#484f58')
        
        # Mark best
        marker = '▲' if higher_better else '▼'
        ax.annotate(f'{marker} {vals[best_idx]:.4f}\n(L{layers[best_idx]})',
                   xy=(best_idx, vals[best_idx]),
                   xytext=(best_idx, vals[best_idx] * (1.15 if higher_better else 0.85)),
                   fontsize=7, color='#f0f6fc', fontweight='bold', ha='center',
                   arrowprops=dict(arrowstyle='->', color='#f0f6fc', lw=1))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, facecolor='#0d1117', bbox_inches='tight')
    plt.close()
    print(f"  Saved summary plot: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', required=True, help='Path to summary.json')
    parser.add_argument('--output_dir', required=True, help='Directory for output plots')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = load_results(args.summary)
    
    if not results:
        print("No results found in summary.json")
        return

    print(f"\nGenerating plots for {len(results)} layers ...")

    make_summary_plot(
        results, os.path.join(args.output_dir, 'ablation_reasoning_summary.png'))

    # Print best layer
    best = max(results, key=lambda r: r.get('accuracy', 0))
    print(f"\n🏆 Best layer (reasoning): {best['extract_layer']}  (Accuracy = {best['accuracy']:.4f})")


if __name__ == '__main__':
    main()
