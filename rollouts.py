"""
Rollout Generation and ELO Tournament for Preference Learning

This script implements Steps 5-6 of the methodology:
- Step 5: Generate k rollouts per test example using high-temperature sampling
- Step 6: Run Bradley-Terry/ELO tournament with the expert judge model

Features:
- Position randomization (A/B swap) to avoid positional bias
- Matchup order randomization to prevent ordering effects
- Adaptive sigmoid squashing for [0,1] reward normalization
- Compatible with Together AI fine-tuned models

Usage:
    python rollout_tournament.py --judge-model <your-finetuned-model-id> --num-questions 10 --num-rollouts 8
"""

import json
import random
import math
import argparse
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

from together import Together
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default models
DEFAULT_GENERATOR_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
DEFAULT_JUDGE_MODEL = "ishanmehta/Meta-Llama-3.1-8B-Instruct-Reference-preference-expert-49eb8762"

# Retry configuration
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2  # seconds


def call_with_retry(func, *args, **kwargs):
    """Call a function with exponential backoff retry logic."""
    last_exception = None
    
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            # Retry on 503 (overloaded) or 429 (rate limit) errors
            if "503" in error_str or "overloaded" in error_str.lower() or "429" in error_str:
                last_exception = e
                delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                print(f"      Retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(delay)
            else:
                raise e
    
    raise last_exception

# Generation settings
DEFAULT_NUM_ROLLOUTS = 8  # k rollouts per question
DEFAULT_TEMPERATURE = 0.8  # High temperature for diversity
DEFAULT_MAX_TOKENS = 300

# Tournament settings
DEFAULT_ELO_K = 32  # ELO K-factor

# Output settings
DEFAULT_OUTPUT_DIR = "data"
REWARDS_OUTPUT_FILE = "rollout_rewards.json"
TRAINING_DATA_FILE = "grpo_training_data.json"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RolloutResult:
    """Result of generating rollouts for a single prompt"""
    prompt_id: str
    prompt: str
    rollouts: List[str]
    elo_ratings: List[float]
    rewards: List[float]  # Sigmoid-squashed to [0,1]
    num_comparisons: int
    

@dataclass  
class TrainingExample:
    """Single training example for GRPO"""
    prompt_id: str
    prompt: str
    response: str
    reward: float
    rank: int  # 1 = best, k = worst


# =============================================================================
# EXPERT JUDGE PROMPT (matches fine-tuning format exactly)
# =============================================================================

SYSTEM_MESSAGE = """You are an expert at analyzing responses. Compare two responses objectively and determine which one is better and why. Consider factors like helpfulness, accuracy, relevance, clarity, and overall usefulness to the person asking the question."""

def format_judge_prompt(prompt: str, response_a: str, response_b: str) -> str:
    """
    Format the prompt for the expert judge model.
    MUST match the exact format used during fine-tuning for best results.
    """
    return f"""Which response is better? Analyze the differences between these two responses.

Original Post:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Think through this step-by-step:
1. First, I'll analyze the helpfulness and relevance...
2. Next, I'll examine the clarity and completeness...
3. Then, I'll evaluate the accuracy and usefulness...
4. Finally, I'll assess the overall quality...

**ANSWER IN THE FOLLOWING FORMAT:**
Response [A/B] is better because..."""


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ultrafeedback_prompts(split: str = "train", sample_size: Optional[int] = None, seed: int = 42) -> List[Dict]:
    """
    Load prompts from the UltraFeedback dataset.
    
    Args:
        split: Dataset split to use
        sample_size: Number of prompts to sample (None = all)
        seed: Random seed for sampling
        
    Returns:
        List of prompt dictionaries
    """
    print(f"Loading UltraFeedback dataset (split: {split})...")
    ds = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")
    dataset = ds[split]
    
    # Extract unique prompts
    prompts = []
    seen_prompts = set()
    
    for i, example in enumerate(dataset):
        prompt_text = example['prompt']
        
        # Deduplicate prompts
        if prompt_text not in seen_prompts:
            seen_prompts.add(prompt_text)
            prompts.append({
                'prompt_id': f"uf_{i}",
                'prompt': prompt_text,
                # Store original responses for reference
                'original_chosen': example['chosen'][1]['content'] if len(example['chosen']) > 1 else "",
                'original_rejected': example['rejected'][1]['content'] if len(example['rejected']) > 1 else "",
            })
    
    print(f"Found {len(prompts)} unique prompts")
    
    # Sample if requested
    if sample_size and sample_size < len(prompts):
        random.seed(seed)
        prompts = random.sample(prompts, sample_size)
        print(f"Sampled {len(prompts)} prompts")
    
    return prompts


# =============================================================================
# ROLLOUT GENERATION (Step 5)
# =============================================================================

def generate_rollouts(
    client: Together,
    prompt: str,
    num_rollouts: int = DEFAULT_NUM_ROLLOUTS,
    generator_model: str = DEFAULT_GENERATOR_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> List[str]:
    """
    Generate multiple diverse responses to a prompt using high-temperature sampling.
    
    Args:
        client: Together AI client
        prompt: The input prompt
        num_rollouts: Number of responses to generate (k)
        generator_model: Model to use for generation
        temperature: Sampling temperature (higher = more diverse)
        max_tokens: Maximum tokens per response
        
    Returns:
        List of generated responses
    """
    rollouts = []
    
    # Format as instruction
    generation_prompt = f"Please provide a helpful response to the following:\n\n{prompt}"
    
    for i in range(num_rollouts):
        try:
            def make_request():
                return client.chat.completions.create(
                    model=generator_model,
                    messages=[
                        {"role": "user", "content": generation_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            
            response = call_with_retry(make_request)
            rollout = response.choices[0].message.content.strip()
            rollouts.append(rollout)
            
        except Exception as e:
            print(f"    Error generating rollout {i+1}: {e}")
            continue
    
    return rollouts


# =============================================================================
# PAIRWISE JUDGING
# =============================================================================

def judge_pair(
    client: Together,
    prompt: str,
    response_a: str,
    response_b: str,
    judge_model: str
) -> Optional[str]:
    """
    Use the fine-tuned expert model to judge which response is better.
    
    Args:
        client: Together AI client
        prompt: Original prompt
        response_a: First response
        response_b: Second response
        judge_model: Fine-tuned judge model ID
        
    Returns:
        "A" or "B" indicating the winner, or None if parsing failed
    """
    judge_prompt = format_judge_prompt(prompt, response_a, response_b)
    
    try:
        def make_request():
            return client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": judge_prompt}
                ],
                max_tokens=300,
                temperature=0  # Deterministic judging
            )
        
        response = call_with_retry(make_request)
        judgment = response.choices[0].message.content.strip()
        
        # Parse the judgment
        if "Response A is better" in judgment:
            return "A"
        elif "Response B is better" in judgment:
            return "B"
        else:
            # Fallback parsing
            if judgment.startswith("Response A") or "prefer A" in judgment.lower():
                return "A"
            elif judgment.startswith("Response B") or "prefer B" in judgment.lower():
                return "B"
            return None
            
    except Exception as e:
        print(f"    Error in judgment: {e}")
        return None


# =============================================================================
# ELO TOURNAMENT (Step 6)
# =============================================================================

def update_elo(rating_a: float, rating_b: float, winner: str, k: float = DEFAULT_ELO_K) -> Tuple[float, float]:
    """
    Update ELO ratings based on match result.
    
    Args:
        rating_a: Current rating of player A
        rating_b: Current rating of player B
        winner: "A", "B", or "tie"
        k: K-factor (sensitivity of updates)
        
    Returns:
        Tuple of (new_rating_a, new_rating_b)
    """
    # Expected scores
    expected_a = 1 / (1 + 10**((rating_b - rating_a) / 400))
    expected_b = 1 - expected_a
    
    # Actual scores
    if winner == "A":
        score_a, score_b = 1.0, 0.0
    elif winner == "B":
        score_a, score_b = 0.0, 1.0
    else:  # Tie
        score_a, score_b = 0.5, 0.5
    
    # Update ratings
    new_rating_a = rating_a + k * (score_a - expected_a)
    new_rating_b = rating_b + k * (score_b - expected_b)
    
    return new_rating_a, new_rating_b


def elo_to_rewards(elo_ratings: List[float], method: str = "minmax") -> List[float]:
    """
    Convert ELO ratings to [0,1] rewards.
    
    Args:
        elo_ratings: List of raw ELO ratings
        method: "minmax" for linear scaling, "sigmoid" for sigmoid with fixed scale
        
    Returns:
        List of rewards in [0,1]
    """
    if len(elo_ratings) == 0:
        return []
    
    if len(elo_ratings) == 1:
        return [0.5]
    
    min_r = min(elo_ratings)
    max_r = max(elo_ratings)
    
    if method == "minmax":
        # Simple min-max normalization to [0, 1]
        spread = max_r - min_r
        if spread == 0:
            return [0.5] * len(elo_ratings)
        return [(r - min_r) / spread for r in elo_ratings]
    
    elif method == "sigmoid":
        # Sigmoid with fixed scale (not adaptive)
        # Scale of 100 means ~400 ELO difference = ~0.98 vs 0.02
        center = sum(elo_ratings) / len(elo_ratings)
        scale = 100
        return [1 / (1 + math.exp(-(r - center) / scale)) for r in elo_ratings]
    
    else:
        raise ValueError(f"Unknown method: {method}")


def run_elo_tournament(
    client: Together,
    prompt: str,
    rollouts: List[str],
    judge_model: str,
    k_factor: float = DEFAULT_ELO_K,
    verbose: bool = True
) -> Tuple[List[float], List[float], int]:
    """
    Run full ELO tournament between all rollouts with randomization.
    
    Features:
    - Randomized matchup order
    - Randomized A/B position presentation
    
    Args:
        client: Together AI client
        prompt: Original prompt
        rollouts: List of generated responses
        judge_model: Fine-tuned judge model ID
        k_factor: ELO K-factor
        verbose: Print progress
        
    Returns:
        Tuple of (elo_ratings, sigmoid_rewards, num_valid_comparisons)
    """
    n = len(rollouts)
    
    # Initialize ELO ratings at 0
    elo_ratings = [0.0] * n
    
    # Create all pairwise comparisons: k(k-1)/2 pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    
    # Randomize matchup order
    random.shuffle(pairs)
    
    total_comparisons = len(pairs)
    valid_comparisons = 0
    
    if verbose:
        print(f"  Running {total_comparisons} pairwise comparisons...")
    
    for pair_idx, (i, j) in enumerate(pairs):
        # Randomly swap A/B positions to avoid positional bias
        if random.random() < 0.5:
            # Normal order: i=A, j=B
            response_a, response_b = rollouts[i], rollouts[j]
            idx_a, idx_b = i, j
        else:
            # Flipped: j=A, i=B
            response_a, response_b = rollouts[j], rollouts[i]
            idx_a, idx_b = j, i
        
        # Get judgment
        winner = judge_pair(client, prompt, response_a, response_b, judge_model)
        
        if winner is not None:
            valid_comparisons += 1
            
            # Determine actual winner index
            if winner == "A":
                winner_idx, loser_idx = idx_a, idx_b
            else:
                winner_idx, loser_idx = idx_b, idx_a
            
            # Update ELO ratings
            new_winner, new_loser = update_elo(
                elo_ratings[winner_idx], 
                elo_ratings[loser_idx], 
                "A",  # Winner is always "A" in update_elo convention
                k=k_factor
            )
            elo_ratings[winner_idx] = new_winner
            elo_ratings[loser_idx] = new_loser
        
        # Progress update
        if verbose and (pair_idx + 1) % 10 == 0:
            print(f"    Completed {pair_idx + 1}/{total_comparisons} comparisons")
    
    # Convert to [0,1] rewards using min-max normalization
    rewards = elo_to_rewards(elo_ratings, method="minmax")
    
    return elo_ratings, rewards, valid_comparisons


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_single_prompt(
    client: Together,
    prompt_data: Dict,
    generator_model: str,
    judge_model: str,
    num_rollouts: int,
    temperature: float,
    verbose: bool = True
) -> Optional[RolloutResult]:
    """
    Process a single prompt: generate rollouts and run tournament.
    
    Args:
        client: Together AI client
        prompt_data: Dictionary with 'prompt_id' and 'prompt' keys
        generator_model: Model for generating rollouts
        judge_model: Fine-tuned expert judge model
        num_rollouts: Number of rollouts to generate
        temperature: Generation temperature
        verbose: Print progress
        
    Returns:
        RolloutResult or None if failed
    """
    prompt_id = prompt_data['prompt_id']
    prompt = prompt_data['prompt']
    
    if verbose:
        print(f"\n  Prompt ID: {prompt_id}")
        print(f"  Prompt: {prompt[:100]}...")
    
    # Step 5: Generate rollouts
    if verbose:
        print(f"  Generating {num_rollouts} rollouts...")
    
    rollouts = generate_rollouts(
        client=client,
        prompt=prompt,
        num_rollouts=num_rollouts,
        generator_model=generator_model,
        temperature=temperature
    )
    
    if len(rollouts) < 2:
        print(f"  ERROR: Only generated {len(rollouts)} rollouts, need at least 2")
        return None
    
    if verbose:
        print(f"  Generated {len(rollouts)} rollouts successfully")
    
    # Step 6: Run ELO tournament
    elo_ratings, rewards, num_comparisons = run_elo_tournament(
        client=client,
        prompt=prompt,
        rollouts=rollouts,
        judge_model=judge_model,
        verbose=verbose
    )
    
    if verbose:
        print(f"  Tournament complete: {num_comparisons} valid comparisons")
        print(f"  Reward range: [{min(rewards):.3f}, {max(rewards):.3f}]")
    
    return RolloutResult(
        prompt_id=prompt_id,
        prompt=prompt,
        rollouts=rollouts,
        elo_ratings=elo_ratings,
        rewards=rewards,
        num_comparisons=num_comparisons
    )


def run_full_pipeline(
    client: Together,
    prompts: List[Dict],
    generator_model: str,
    judge_model: str,
    num_rollouts: int = DEFAULT_NUM_ROLLOUTS,
    temperature: float = DEFAULT_TEMPERATURE,
    verbose: bool = True
) -> Tuple[List[RolloutResult], List[TrainingExample]]:
    """
    Run the full rollout generation and tournament pipeline on multiple prompts.
    
    Args:
        client: Together AI client
        prompts: List of prompt dictionaries
        generator_model: Model for generation
        judge_model: Fine-tuned judge model
        num_rollouts: Rollouts per prompt
        temperature: Generation temperature
        verbose: Print progress
        
    Returns:
        Tuple of (all_results, training_examples)
    """
    all_results = []
    training_examples = []
    
    print(f"\n{'='*60}")
    print(f"ROLLOUT GENERATION AND TOURNAMENT PIPELINE")
    print(f"{'='*60}")
    print(f"Generator model: {generator_model}")
    print(f"Judge model: {judge_model}")
    print(f"Num prompts: {len(prompts)}")
    print(f"Rollouts per prompt: {num_rollouts}")
    print(f"Temperature: {temperature}")
    print(f"Expected comparisons per prompt: {num_rollouts * (num_rollouts - 1) // 2}")
    print(f"{'='*60}")
    
    for i, prompt_data in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Processing prompt...")
        
        result = process_single_prompt(
            client=client,
            prompt_data=prompt_data,
            generator_model=generator_model,
            judge_model=judge_model,
            num_rollouts=num_rollouts,
            temperature=temperature,
            verbose=verbose
        )
        
        if result is not None:
            all_results.append(result)
            
            # Create training examples with ranks
            # Sort by reward to get ranks (1 = best)
            indexed_rewards = list(enumerate(result.rewards))
            sorted_by_reward = sorted(indexed_rewards, key=lambda x: x[1], reverse=True)
            
            for rank, (orig_idx, reward) in enumerate(sorted_by_reward, start=1):
                example = TrainingExample(
                    prompt_id=result.prompt_id,
                    prompt=result.prompt,
                    response=result.rollouts[orig_idx],
                    reward=reward,
                    rank=rank
                )
                training_examples.append(example)
    
    # Print summary statistics
    print_summary_statistics(all_results, training_examples)
    
    return all_results, training_examples


def print_summary_statistics(results: List[RolloutResult], training_examples: List[TrainingExample]):
    """Print summary statistics for the pipeline run."""
    
    if not results:
        print("\nNo results to summarize.")
        return
    
    all_rewards = [ex.reward for ex in training_examples]
    
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Prompts processed: {len(results)}")
    print(f"Total training examples: {len(training_examples)}")
    print(f"\nReward Statistics:")
    print(f"  Min:  {min(all_rewards):.4f}")
    print(f"  Max:  {max(all_rewards):.4f}")
    print(f"  Mean: {sum(all_rewards)/len(all_rewards):.4f}")
    
    # Distribution buckets (for [0,1] rewards)
    high_quality = sum(1 for r in all_rewards if r > 0.7)
    medium_quality = sum(1 for r in all_rewards if 0.3 <= r <= 0.7)
    low_quality = sum(1 for r in all_rewards if r < 0.3)
    
    print(f"\nQuality Distribution:")
    print(f"  High (>0.7):    {high_quality} ({high_quality/len(all_rewards)*100:.1f}%)")
    print(f"  Medium (0.3-0.7): {medium_quality} ({medium_quality/len(all_rewards)*100:.1f}%)")
    print(f"  Low (<0.3):     {low_quality} ({low_quality/len(all_rewards)*100:.1f}%)")
    
    # Comparison success rate
    total_expected = sum(len(r.rollouts) * (len(r.rollouts) - 1) // 2 for r in results)
    total_actual = sum(r.num_comparisons for r in results)
    print(f"\nComparison Success Rate: {total_actual}/{total_expected} ({total_actual/total_expected*100:.1f}%)")
    print(f"{'='*60}")


# =============================================================================
# FILE I/O
# =============================================================================

def save_results(
    results: List[RolloutResult],
    training_examples: List[TrainingExample],
    output_dir: str = DEFAULT_OUTPUT_DIR
):
    """Save results and training data to JSON files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    results_file = output_path / REWARDS_OUTPUT_FILE
    results_data = [asdict(r) for r in results]
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nSaved full results to: {results_file}")
    
    # Save training examples (for GRPO)
    training_file = output_path / TRAINING_DATA_FILE
    training_data = [asdict(ex) for ex in training_examples]
    with open(training_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"Saved training data to: {training_file}")
    
    return results_file, training_file


def load_results(filepath: str) -> List[Dict]:
    """Load previously saved results."""
    with open(filepath, 'r') as f:
        return json.load(f)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate rollouts and run ELO tournament for preference learning"
    )
    
    # Required arguments
    parser.add_argument(
        "--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
        help=f"Fine-tuned judge model ID (default: {DEFAULT_JUDGE_MODEL})"
    )
    
    # Optional arguments
    parser.add_argument(
        "--generator-model", type=str, default=DEFAULT_GENERATOR_MODEL,
        help=f"Model for generating rollouts (default: {DEFAULT_GENERATOR_MODEL})"
    )
    parser.add_argument(
        "--num-questions", type=int, default=10,
        help="Number of prompts to process (default: 10)"
    )
    parser.add_argument(
        "--num-rollouts", type=int, default=DEFAULT_NUM_ROLLOUTS,
        help=f"Rollouts per prompt (default: {DEFAULT_NUM_ROLLOUTS})"
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE,
        help=f"Generation temperature (default: {DEFAULT_TEMPERATURE})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Initialize client
    client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
    
    # Load prompts from UltraFeedback
    prompts = load_ultrafeedback_prompts(
        split="train",
        sample_size=args.num_questions * 2,  # Load extra for filtering
        seed=args.seed
    )
    
    # Take requested number
    prompts = prompts[:args.num_questions]
    
    # Run pipeline
    results, training_examples = run_full_pipeline(
        client=client,
        prompts=prompts,
        generator_model=args.generator_model,
        judge_model=args.judge_model,
        num_rollouts=args.num_rollouts,
        temperature=args.temperature,
        verbose=not args.quiet
    )
    
    # Save results
    save_results(results, training_examples, args.output_dir)
    
    print(f"\n✓ Pipeline complete!")
    print(f"  Training examples ready for GRPO: {len(training_examples)}")
    print(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()