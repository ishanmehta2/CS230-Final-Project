"""
Rollout Analysis and Rubric Scoring

This script:
1. Loads rollout results from the tournament
2. Scores each response on a multi-dimensional rubric using an LLM
3. Compares rubric scores to ELO-derived rewards
4. Generates analysis and visualizations

Usage:
    python analyze_rollouts.py --input data/rollout_rewards.json
    python analyze_rollouts.py --input data/rollout_rewards.json --score-model gpt-4o
"""

import json
import argparse
import os
import random
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import statistics

from together import Together
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_INPUT_FILE = "data/rollout_rewards.json"
DEFAULT_SCORE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
DEFAULT_OUTPUT_FILE = "data/rollout_analysis.json"

# Rubric dimensions (1-5 scale each)
RUBRIC_DIMENSIONS = [
    "helpfulness",
    "accuracy", 
    "clarity",
    "completeness",
    "relevance"
]


# =============================================================================
# RUBRIC SCORING
# =============================================================================

def format_rubric_prompt(prompt: str, response: str) -> str:
    """Format prompt for rubric-based scoring."""
    return f"""You are an expert evaluator. Score the following response on a 1-5 scale for each dimension.

**Original Prompt:**
{prompt}

**Response to Evaluate:**
{response}

**Scoring Rubric (1-5 for each):**
- **Helpfulness**: Does it address the user's needs? (1=not helpful, 5=extremely helpful)
- **Accuracy**: Is the information correct? (1=many errors, 5=fully accurate)
- **Clarity**: Is it well-written and easy to understand? (1=confusing, 5=crystal clear)
- **Completeness**: Does it fully address the question? (1=incomplete, 5=comprehensive)
- **Relevance**: Does it stay on topic? (1=off-topic, 5=perfectly relevant)

**Respond in this EXACT JSON format (no other text):**
{{"helpfulness": X, "accuracy": X, "clarity": X, "completeness": X, "relevance": X}}

Where X is an integer from 1-5."""


def score_response_with_rubric(
    client: Together,
    prompt: str,
    response: str,
    model: str = DEFAULT_SCORE_MODEL
) -> Optional[Dict[str, int]]:
    """Score a single response using the rubric."""
    
    rubric_prompt = format_rubric_prompt(prompt, response)
    
    try:
        api_response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": rubric_prompt}],
            max_tokens=100,
            temperature=0
        )
        
        content = api_response.choices[0].message.content.strip()
        
        # Parse JSON response
        # Handle potential markdown code blocks
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        scores = json.loads(content)
        
        # Validate scores
        for dim in RUBRIC_DIMENSIONS:
            if dim not in scores:
                print(f"    Missing dimension: {dim}")
                return None
            if not isinstance(scores[dim], (int, float)) or scores[dim] < 1 or scores[dim] > 5:
                print(f"    Invalid score for {dim}: {scores[dim]}")
                return None
        
        return scores
        
    except json.JSONDecodeError as e:
        print(f"    JSON parse error: {e}")
        print(f"    Raw response: {content[:200]}")
        return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def compute_rubric_total(scores: Dict[str, int]) -> float:
    """Compute total rubric score (normalized to 0-1)."""
    total = sum(scores[dim] for dim in RUBRIC_DIMENSIONS)
    max_possible = 5 * len(RUBRIC_DIMENSIONS)  # 25
    return total / max_possible


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_correlation(x: List[float], y: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    
    var_x = sum((xi - mean_x) ** 2 for xi in x)
    var_y = sum((yi - mean_y) ** 2 for yi in y)
    
    denominator = (var_x * var_y) ** 0.5
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def analyze_results(scored_data: List[Dict]) -> Dict:
    """Compute analysis statistics."""
    
    elo_rewards = [d['elo_reward'] for d in scored_data]
    rubric_totals = [d['rubric_total'] for d in scored_data]
    
    # Correlation between ELO rewards and rubric scores
    correlation = compute_correlation(elo_rewards, rubric_totals)
    
    # Per-dimension correlations
    dim_correlations = {}
    for dim in RUBRIC_DIMENSIONS:
        dim_scores = [d['rubric_scores'][dim] for d in scored_data]
        dim_correlations[dim] = compute_correlation(elo_rewards, dim_scores)
    
    # Find disagreements (high ELO but low rubric, or vice versa)
    disagreements = []
    for d in scored_data:
        elo_rank = "high" if d['elo_reward'] > 0.7 else ("low" if d['elo_reward'] < 0.3 else "mid")
        rubric_rank = "high" if d['rubric_total'] > 0.7 else ("low" if d['rubric_total'] < 0.3 else "mid")
        
        if (elo_rank == "high" and rubric_rank == "low") or (elo_rank == "low" and rubric_rank == "high"):
            disagreements.append({
                'prompt_id': d['prompt_id'],
                'elo_reward': d['elo_reward'],
                'rubric_total': d['rubric_total'],
                'elo_rank': elo_rank,
                'rubric_rank': rubric_rank,
                'response_preview': d['response'][:200] + "..."
            })
    
    # Average rubric scores per dimension
    avg_dim_scores = {}
    for dim in RUBRIC_DIMENSIONS:
        scores = [d['rubric_scores'][dim] for d in scored_data]
        avg_dim_scores[dim] = sum(scores) / len(scores)
    
    # High vs Low ELO comparison
    high_elo = [d for d in scored_data if d['elo_reward'] > 0.7]
    low_elo = [d for d in scored_data if d['elo_reward'] < 0.3]
    
    high_elo_rubric = sum(d['rubric_total'] for d in high_elo) / len(high_elo) if high_elo else 0
    low_elo_rubric = sum(d['rubric_total'] for d in low_elo) / len(low_elo) if low_elo else 0
    
    return {
        'overall_correlation': correlation,
        'dimension_correlations': dim_correlations,
        'avg_dimension_scores': avg_dim_scores,
        'high_elo_avg_rubric': high_elo_rubric,
        'low_elo_avg_rubric': low_elo_rubric,
        'rubric_gap': high_elo_rubric - low_elo_rubric,
        'num_disagreements': len(disagreements),
        'disagreements': disagreements[:5],  # Top 5
        'total_scored': len(scored_data)
    }


def print_analysis_report(analysis: Dict):
    """Print a formatted analysis report."""
    
    print(f"\n{'='*70}")
    print("ROLLOUT ANALYSIS REPORT")
    print(f"{'='*70}")
    
    print(f"\n📊 CORRELATION ANALYSIS")
    print(f"   Overall ELO-Rubric Correlation: {analysis['overall_correlation']:.3f}")
    
    # Interpret correlation
    corr = analysis['overall_correlation']
    if corr > 0.7:
        interpretation = "Strong positive - ELO rewards align well with rubric quality!"
    elif corr > 0.4:
        interpretation = "Moderate positive - Reasonable alignment"
    elif corr > 0.1:
        interpretation = "Weak positive - Some alignment but noisy"
    elif corr > -0.1:
        interpretation = "No correlation - ELO and rubric measure different things"
    else:
        interpretation = "Negative - ELO rewards may be inversely related to quality!"
    print(f"   Interpretation: {interpretation}")
    
    print(f"\n📏 PER-DIMENSION CORRELATIONS (with ELO reward):")
    for dim, corr in sorted(analysis['dimension_correlations'].items(), key=lambda x: -x[1]):
        bar = "█" * int(abs(corr) * 20)
        sign = "+" if corr >= 0 else "-"
        print(f"   {dim:15s}: {sign}{abs(corr):.3f} {bar}")
    
    print(f"\n📈 AVERAGE RUBRIC SCORES (1-5 scale):")
    for dim, score in sorted(analysis['avg_dimension_scores'].items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 4)
        print(f"   {dim:15s}: {score:.2f}/5.0 {bar}")
    
    print(f"\n🎯 HIGH vs LOW ELO QUALITY GAP:")
    print(f"   High ELO (>0.7) avg rubric: {analysis['high_elo_avg_rubric']:.3f}")
    print(f"   Low ELO (<0.3) avg rubric:  {analysis['low_elo_avg_rubric']:.3f}")
    print(f"   Gap: {analysis['rubric_gap']:.3f}")
    
    if analysis['rubric_gap'] > 0.1:
        print(f"   ✓ Good! High-ELO responses have higher rubric scores.")
    elif analysis['rubric_gap'] > 0:
        print(f"   ~ Marginal difference - judge may need improvement")
    else:
        print(f"   ✗ Problem! High-ELO responses have LOWER rubric scores.")
    
    print(f"\n⚠️  DISAGREEMENTS (ELO vs Rubric):")
    print(f"   Found {analysis['num_disagreements']} cases where ELO and rubric disagree")
    
    if analysis['disagreements']:
        print(f"\n   Sample disagreements:")
        for i, d in enumerate(analysis['disagreements'][:3], 1):
            print(f"\n   [{i}] Prompt: {d['prompt_id']}")
            print(f"       ELO: {d['elo_reward']:.2f} ({d['elo_rank']})")
            print(f"       Rubric: {d['rubric_total']:.2f} ({d['rubric_rank']})")
            print(f"       Response: {d['response_preview'][:100]}...")
    
    print(f"\n{'='*70}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze rollouts with rubric scoring")
    
    parser.add_argument(
        "--input", type=str, default=DEFAULT_INPUT_FILE,
        help=f"Input rollout results file (default: {DEFAULT_INPUT_FILE})"
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT_FILE,
        help=f"Output analysis file (default: {DEFAULT_OUTPUT_FILE})"
    )
    parser.add_argument(
        "--score-model", type=str, default=DEFAULT_SCORE_MODEL,
        help=f"Model for rubric scoring (default: {DEFAULT_SCORE_MODEL})"
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Number of rollouts to score (default: all)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--skip-scoring", action="store_true",
        help="Skip scoring, just analyze existing data"
    )
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Load rollout results
    print(f"Loading rollout results from {args.input}...")
    with open(args.input, 'r') as f:
        rollout_results = json.load(f)
    
    print(f"Loaded {len(rollout_results)} prompts")
    
    # Flatten to individual rollouts
    all_rollouts = []
    for result in rollout_results:
        for i, (rollout, reward) in enumerate(zip(result['rollouts'], result['rewards'])):
            all_rollouts.append({
                'prompt_id': result['prompt_id'],
                'prompt': result['prompt'],
                'response': rollout,
                'elo_reward': reward,
                'rollout_idx': i
            })
    
    print(f"Total rollouts: {len(all_rollouts)}")
    
    # Sample if requested
    if args.sample_size and args.sample_size < len(all_rollouts):
        all_rollouts = random.sample(all_rollouts, args.sample_size)
        print(f"Sampled {len(all_rollouts)} rollouts")
    
    # Score with rubric
    if not args.skip_scoring:
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        
        print(f"\n{'='*60}")
        print(f"RUBRIC SCORING")
        print(f"{'='*60}")
        print(f"Model: {args.score_model}")
        print(f"Dimensions: {', '.join(RUBRIC_DIMENSIONS)}")
        print(f"{'='*60}")
        
        scored_data = []
        
        for i, rollout in enumerate(all_rollouts):
            print(f"\n[{i+1}/{len(all_rollouts)}] Scoring {rollout['prompt_id']} rollout {rollout['rollout_idx']}...")
            
            scores = score_response_with_rubric(
                client=client,
                prompt=rollout['prompt'],
                response=rollout['response'],
                model=args.score_model
            )
            
            if scores:
                rubric_total = compute_rubric_total(scores)
                scored_data.append({
                    **rollout,
                    'rubric_scores': scores,
                    'rubric_total': rubric_total
                })
                print(f"   Rubric: {rubric_total:.2f} | ELO: {rollout['elo_reward']:.2f}")
            else:
                print(f"   Failed to score, skipping")
        
        print(f"\nSuccessfully scored {len(scored_data)}/{len(all_rollouts)} rollouts")
        
    else:
        # Load existing scored data
        print("Skipping scoring, loading existing data...")
        with open(args.output, 'r') as f:
            data = json.load(f)
        scored_data = data.get('scored_rollouts', [])
    
    if not scored_data:
        print("No scored data to analyze!")
        return
    
    # Run analysis
    print(f"\nRunning analysis...")
    analysis = analyze_results(scored_data)
    
    # Print report
    print_analysis_report(analysis)
    
    # Save results
    output_data = {
        'scored_rollouts': scored_data,
        'analysis': analysis
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Analysis saved to: {args.output}")


if __name__ == "__main__":
    main()
