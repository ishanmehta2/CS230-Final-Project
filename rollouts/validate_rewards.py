import json
import argparse
import numpy as np
import re

# this is for automatic signals
def compute_quality_signals(text):
    return {
        'length': len(text),
        'word_count': len(text.split()),
        'has_code': 1 if '```' in text or 'python' in text.lower() else 0,
        'has_explanation': 1 if any(w in text.lower() for w in ['because', 'therefore', 'thus', 'since', 'explanation']) else 0,
        'has_steps': 1 if any(p in text for p in ['1.', '1)', 'Step 1', 'First,']) else 0,
        'sentence_count': len(re.split(r'[.!?]+', text)),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/rollout_rewards.json")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    # Collect all rollouts with rewards
    all_rollouts = []
    for item in data:
        for i, (rollout, reward) in enumerate(zip(item['rollouts'], item['rewards'])):
            all_rollouts.append({
                'prompt_id': item['prompt_id'],
                'prompt': item['prompt'][:100] + '...',
                'rollout': rollout,
                'reward': reward,
                'signals': compute_quality_signals(rollout)
            })

    # Sort by reward
    all_rollouts.sort(key=lambda x: x['reward'], reverse=True)

    print("=" * 70)
    print("REWARD VALIDATION ANALYSIS")
    print("=" * 70)

    # 1. Quality proxy correlations
    print("\n1. CORRELATION WITH QUALITY PROXIES")
    print("-" * 50)
    
    rewards = np.array([r['reward'] for r in all_rollouts])
    
    for signal_name in ['length', 'word_count', 'has_explanation', 'has_steps']:
        signal_values = np.array([r['signals'][signal_name] for r in all_rollouts])
        corr = np.corrcoef(rewards, signal_values)[0, 1]
        print(f"  {signal_name:20s}: r = {corr:+.3f}")

    # 2. High vs Low reward statistics
    print("\n2. HIGH vs LOW REWARD COMPARISON")
    print("-" * 50)
    
    n = len(all_rollouts)
    top_25 = all_rollouts[:n//4]
    bottom_25 = all_rollouts[-n//4:]

    print(f"  {'Metric':<20s} {'Top 25%':>12s} {'Bottom 25%':>12s}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")
    
    for signal in ['word_count', 'has_explanation', 'has_steps']:
        top_avg = np.mean([r['signals'][signal] for r in top_25])
        bot_avg = np.mean([r['signals'][signal] for r in bottom_25])
        print(f"  {signal:<20s} {top_avg:>12.1f} {bot_avg:>12.1f}")

    # 3. Example comparison
    print("\n3. QUALITATIVE EXAMPLES")
    print("-" * 50)
    
    print("\n>>> HIGH REWARD EXAMPLE (reward = {:.2f})".format(all_rollouts[0]['reward']))
    print(f"Prompt: {all_rollouts[0]['prompt']}")
    print(f"Response: {all_rollouts[0]['rollout'][:500]}...")
    
    print("\n>>> LOW REWARD EXAMPLE (reward = {:.2f})".format(all_rollouts[-1]['reward']))
    print(f"Prompt: {all_rollouts[-1]['prompt']}")
    print(f"Response: {all_rollouts[-1]['rollout'][:500]}...")

    # 4. Summary statistics
    print("\n4. REWARD DISTRIBUTION SUMMARY")
    print("-" * 50)
    print(f"  Total rollouts: {len(all_rollouts)}")
    print(f"  Mean reward:    {np.mean(rewards):.3f}")
    print(f"  Std reward:     {np.std(rewards):.3f}")
    print(f"  High (>0.7):    {sum(r > 0.7 for r in rewards)} ({100*sum(r > 0.7 for r in rewards)/len(rewards):.1f}%)")
    print(f"  Medium:         {sum(0.3 <= r <= 0.7 for r in rewards)} ({100*sum(0.3 <= r <= 0.7 for r in rewards)/len(rewards):.1f}%)")
    print(f"  Low (<0.3):     {sum(r < 0.3 for r in rewards)} ({100*sum(r < 0.3 for r in rewards)/len(rewards):.1f}%)")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("  - Positive correlation with explanation/steps suggests rewards")
    print("    capture response quality, not just length")
    print("  - Top quartile should show more structured, complete responses")
    print("  - Bottom quartile should show terse or incomplete responses")
    print("=" * 70)

if __name__ == "__main__":
    main()
