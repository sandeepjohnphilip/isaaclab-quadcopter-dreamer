"""
COMPARE PPO vs DREAMERV2 TRAINING CURVES
=========================================

Creates comparison plots from training logs.

Usage:
    python compare_training.py --ppo ppo_training_log.json --dreamer dreamer_training_log.json --output comparison.png
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np

def smooth(data, window=3):
    """Simple moving average smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def load_log(filepath):
    """Load training log from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_comparison(ppo_log, dreamer_log, output_path, title_suffix=""):
    """Create comparison plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'PPO vs DreamerV2 Comparison{title_suffix}', fontsize=14, fontweight='bold')
    
    # Colors
    ppo_color = '#2ecc71'      # Green
    dreamer_color = '#3498db'  # Blue
    
    # 1. Success Rate
    ax = axes[0, 0]
    if ppo_log:
        ax.plot(ppo_log['steps'], ppo_log['success_rate'], 
                color=ppo_color, alpha=0.3, linewidth=1)
        if len(ppo_log['success_rate']) >= 3:
            smoothed = smooth(ppo_log['success_rate'])
            ax.plot(ppo_log['steps'][:len(smoothed)], smoothed, 
                    color=ppo_color, linewidth=2, label='PPO')
    if dreamer_log:
        ax.plot(dreamer_log['steps'], dreamer_log['success_rate'], 
                color=dreamer_color, alpha=0.3, linewidth=1)
        if len(dreamer_log['success_rate']) >= 3:
            smoothed = smooth(dreamer_log['success_rate'])
            ax.plot(dreamer_log['steps'][:len(smoothed)], smoothed, 
                    color=dreamer_color, linewidth=2, label='DreamerV2')
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Success Rate Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    
    # 2. Average Distance
    ax = axes[0, 1]
    if ppo_log:
        ax.plot(ppo_log['steps'], ppo_log['avg_distance'], 
                color=ppo_color, alpha=0.3, linewidth=1)
        if len(ppo_log['avg_distance']) >= 3:
            smoothed = smooth(ppo_log['avg_distance'])
            ax.plot(ppo_log['steps'][:len(smoothed)], smoothed, 
                    color=ppo_color, linewidth=2, label='PPO')
    if dreamer_log:
        ax.plot(dreamer_log['steps'], dreamer_log['avg_distance'], 
                color=dreamer_color, alpha=0.3, linewidth=1)
        if len(dreamer_log['avg_distance']) >= 3:
            smoothed = smooth(dreamer_log['avg_distance'])
            ax.plot(dreamer_log['steps'][:len(smoothed)], smoothed, 
                    color=dreamer_color, linewidth=2, label='DreamerV2')
    ax.axhline(y=9.3, color='red', linestyle='--', alpha=0.5, label='Goal (9.3m)')
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Average Distance (m)')
    ax.set_title('Average Distance Achieved')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Crash Rate
    ax = axes[1, 0]
    if ppo_log:
        ax.plot(ppo_log['steps'], ppo_log['crash_rate'], 
                color=ppo_color, alpha=0.3, linewidth=1)
        if len(ppo_log['crash_rate']) >= 3:
            smoothed = smooth(ppo_log['crash_rate'])
            ax.plot(ppo_log['steps'][:len(smoothed)], smoothed, 
                    color=ppo_color, linewidth=2, label='PPO')
    if dreamer_log:
        ax.plot(dreamer_log['steps'], dreamer_log['crash_rate'], 
                color=dreamer_color, alpha=0.3, linewidth=1)
        if len(dreamer_log['crash_rate']) >= 3:
            smoothed = smooth(dreamer_log['crash_rate'])
            ax.plot(dreamer_log['steps'][:len(smoothed)], smoothed, 
                    color=dreamer_color, linewidth=2, label='DreamerV2')
    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Crash Rate (%)')
    ax.set_title('Crash Rate Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    
    # 4. Sample Efficiency - Steps to reach thresholds
    ax = axes[1, 1]
    
    thresholds = [50, 70, 90]
    ppo_steps_to_threshold = []
    dreamer_steps_to_threshold = []
    
    for thresh in thresholds:
        # PPO
        ppo_step = None
        if ppo_log:
            for i, sr in enumerate(ppo_log['success_rate']):
                if sr >= thresh:
                    ppo_step = ppo_log['steps'][i]
                    break
        ppo_steps_to_threshold.append(ppo_step if ppo_step else float('inf'))
        
        # DreamerV2
        dreamer_step = None
        if dreamer_log:
            for i, sr in enumerate(dreamer_log['success_rate']):
                if sr >= thresh:
                    dreamer_step = dreamer_log['steps'][i]
                    break
        dreamer_steps_to_threshold.append(dreamer_step if dreamer_step else float('inf'))
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    # Replace inf with max steps for display
    max_steps = 0
    if ppo_log:
        max_steps = max(max_steps, max(ppo_log['steps']))
    if dreamer_log:
        max_steps = max(max_steps, max(dreamer_log['steps']))
    
    ppo_display = [s if s != float('inf') else max_steps for s in ppo_steps_to_threshold]
    dreamer_display = [s if s != float('inf') else max_steps for s in dreamer_steps_to_threshold]
    
    bars1 = ax.bar(x - width/2, ppo_display, width, label='PPO', color=ppo_color)
    bars2 = ax.bar(x + width/2, dreamer_display, width, label='DreamerV2', color=dreamer_color)
    
    ax.set_xlabel('Success Rate Threshold (%)')
    ax.set_ylabel('Steps to Reach')
    ax.set_title('Sample Efficiency (Steps to Reach Threshold)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t}%' for t in thresholds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add "Not reached" labels
    for i, (p, d) in enumerate(zip(ppo_steps_to_threshold, dreamer_steps_to_threshold)):
        if p == float('inf'):
            ax.annotate('N/A', (i - width/2, ppo_display[i]), ha='center', va='bottom', fontsize=8)
        if d == float('inf'):
            ax.annotate('N/A', (i + width/2, dreamer_display[i]), ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison plot to: {output_path}")
    plt.close()


def print_summary(ppo_log, dreamer_log):
    """Print summary statistics."""
    
    print("\n" + "="*60)
    print("TRAINING COMPARISON SUMMARY")
    print("="*60)
    
    headers = ["Metric", "PPO", "DreamerV2"]
    rows = []
    
    # Best success rate
    ppo_best = max(ppo_log['success_rate']) if ppo_log else 0
    dreamer_best = max(dreamer_log['success_rate']) if dreamer_log else 0
    rows.append(["Best Success Rate", f"{ppo_best:.1f}%", f"{dreamer_best:.1f}%"])
    
    # Final success rate
    ppo_final = ppo_log['success_rate'][-1] if ppo_log else 0
    dreamer_final = dreamer_log['success_rate'][-1] if dreamer_log else 0
    rows.append(["Final Success Rate", f"{ppo_final:.1f}%", f"{dreamer_final:.1f}%"])
    
    # Steps to 50% success
    def steps_to_threshold(log, thresh):
        if not log:
            return "N/A"
        for i, sr in enumerate(log['success_rate']):
            if sr >= thresh:
                return f"{log['steps'][i]:,}"
        return "Not reached"
    
    rows.append(["Steps to 50% Success", steps_to_threshold(ppo_log, 50), steps_to_threshold(dreamer_log, 50)])
    rows.append(["Steps to 80% Success", steps_to_threshold(ppo_log, 80), steps_to_threshold(dreamer_log, 80)])
    
    # Best distance
    ppo_best_dist = max(ppo_log['avg_distance']) if ppo_log else 0
    dreamer_best_dist = max(dreamer_log['avg_distance']) if dreamer_log else 0
    rows.append(["Best Avg Distance", f"{ppo_best_dist:.2f}m", f"{dreamer_best_dist:.2f}m"])
    
    # Total steps
    ppo_steps = ppo_log['steps'][-1] if ppo_log else 0
    dreamer_steps = dreamer_log['steps'][-1] if dreamer_log else 0
    rows.append(["Total Steps", f"{ppo_steps:,}", f"{dreamer_steps:,}"])
    
    # Print table
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(3)]
    
    print("\n" + "-".join("-" * (w + 2) for w in col_widths))
    print(" | ".join(f"{headers[i]:<{col_widths[i]}}" for i in range(3)))
    print("-".join("-" * (w + 2) for w in col_widths))
    for row in rows:
        print(" | ".join(f"{row[i]:<{col_widths[i]}}" for i in range(3)))
    print("-".join("-" * (w + 2) for w in col_widths))
    
    # Winner
    print("\n📊 ANALYSIS:")
    if ppo_best > dreamer_best:
        print(f"   ✓ PPO achieved higher success rate ({ppo_best:.1f}% vs {dreamer_best:.1f}%)")
    elif dreamer_best > ppo_best:
        print(f"   ✓ DreamerV2 achieved higher success rate ({dreamer_best:.1f}% vs {ppo_best:.1f}%)")
    else:
        print(f"   = Both achieved same best success rate ({ppo_best:.1f}%)")
    
    ppo_50 = steps_to_threshold(ppo_log, 50)
    dreamer_50 = steps_to_threshold(dreamer_log, 50)
    
    if ppo_50 != "Not reached" and dreamer_50 != "Not reached" and ppo_50 != "N/A" and dreamer_50 != "N/A":
        ppo_val = int(ppo_50.replace(",", ""))
        dreamer_val = int(dreamer_50.replace(",", ""))
        if ppo_val < dreamer_val:
            ratio = dreamer_val / ppo_val
            print(f"   ✓ PPO reached 50% success {ratio:.1f}x faster")
        elif dreamer_val < ppo_val:
            ratio = ppo_val / dreamer_val
            print(f"   ✓ DreamerV2 reached 50% success {ratio:.1f}x faster")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo", type=str, help="PPO training log JSON file")
    parser.add_argument("--dreamer", type=str, help="DreamerV2 training log JSON file")
    parser.add_argument("--output", type=str, default="comparison.png", help="Output image path")
    parser.add_argument("--title", type=str, default="", help="Additional title text")
    args = parser.parse_args()
    
    ppo_log = None
    dreamer_log = None
    
    if args.ppo:
        try:
            ppo_log = load_log(args.ppo)
            print(f"✓ Loaded PPO log: {args.ppo}")
        except Exception as e:
            print(f"⚠ Could not load PPO log: {e}")
    
    if args.dreamer:
        try:
            dreamer_log = load_log(args.dreamer)
            print(f"✓ Loaded DreamerV2 log: {args.dreamer}")
        except Exception as e:
            print(f"⚠ Could not load DreamerV2 log: {e}")
    
    if not ppo_log and not dreamer_log:
        print("❌ No logs provided. Use --ppo and/or --dreamer")
        return
    
    title_suffix = f" - {args.title}" if args.title else ""
    plot_comparison(ppo_log, dreamer_log, args.output, title_suffix)
    print_summary(ppo_log, dreamer_log)


if __name__ == "__main__":
    main()
