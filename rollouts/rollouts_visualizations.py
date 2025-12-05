import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/rollout_rewards.json")
    parser.add_argument("--output", default="reward_analysis.png")
    args = parser.parse_args()

    # Load data
    with open(args.input) as f:
        data = json.load(f)
    all_elos = []
    all_rewards = []
    for item in data:
        all_elos.extend(item['elo_ratings'])
        all_rewards.extend(item['rewards'])

    all_elos = np.array(all_elos)
    all_rewards = np.array(all_rewards)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot 1: ELO vs Reward correlation
    ax1 = axes[0]
    ax1.scatter(all_elos, all_rewards, alpha=0.6, s=30)
    z = np.polyfit(all_elos, all_rewards, 1)
    p = np.poly1d(z)
    x_line = np.linspace(all_elos.min(), all_elos.max(), 100)
    ax1.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'r={np.corrcoef(all_elos, all_rewards)[0,1]:.3f}')
    ax1.set_xlabel('ELO Rating')
    ax1.set_ylabel('Normalized Reward')
    ax1.set_title('ELO vs Reward Correlation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Reward distribution
    ax2 = axes[1]
    ax2.hist(all_rewards, bins=15, edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(all_rewards), color='r', linestyle='--', label=f'Mean: {np.mean(all_rewards):.2f}')
    ax2.set_xlabel('Normalized Reward')
    ax2.set_ylabel('Count')
    ax2.set_title('Reward Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved: {args.output}")

    # Print stats
    print(f"\nStats: n={len(all_rewards)}, mean={np.mean(all_rewards):.3f}, std={np.std(all_rewards):.3f}")
    print(f"Correlation (ELO, Reward): {np.corrcoef(all_elos, all_rewards)[0,1]:.4f}")

if __name__ == "__main__":
    main()
