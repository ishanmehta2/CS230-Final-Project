import json
import random
import math
import argparse
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DEFAULT_GENERATOR_MODEL = "gpt-4o-mini"
DEFAULT_JUDGE_MODEL = "ft:gpt-4o-mini-2024-07-18:nimbic-ai:preference-expert:Cix41j5l"
DEFAULT_SPLITS_FILE = "data/ultrafeedback_splits.json"
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2  


def call_with_retry(func, *args, **kwargs):
    last_exception = None
    
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            # Retry on rate limit errors
            if "rate_limit" in error_str.lower() or "429" in error_str:
                last_exception = e
                delay = INITIAL_RETRY_DELAY * (2 ** attempt)
                print(f"      Retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(delay)
            else:
                raise e
    
    raise last_exception

DEFAULT_NUM_ROLLOUTS = 8  
DEFAULT_TEMPERATURE = 0.8  # could use 0.9, but went with 0.8 because wanted some familiarity in answers
DEFAULT_MAX_TOKENS = 300
DEFAULT_ELO_K = 32  # ELO K-factor
DEFAULT_OUTPUT_DIR = "data"
REWARDS_OUTPUT_FILE = "rollout_rewards.json"
TRAINING_DATA_FILE = "grpo_training_data.json"

@dataclass
class RolloutResult:
    """Result of generating rollouts for a single prompt"""
    prompt_id: str
    prompt: str
    rollouts: List[str]
    elo_ratings: List[float]
    rewards: List[float]  # Normalized to [0,1]
    num_comparisons: int
    

@dataclass  
class TrainingExample:
    """Single training example for GRPO"""
    prompt_id: str
    prompt: str
    response: str
    reward: float
    rank: int  # 1 = best, k = worst


# this matches the fine-tuning prompt exactly which is important to ensure the model is seeing the same type of comparisons it was trained on
SYSTEM_MESSAGE = """You are an expert at analyzing responses to questions and instructions. Compare two responses objectively and determine which one is better and why. Consider factors like helpfulness, accuracy, relevance, clarity, and overall usefulness to the person asking the question."""

def format_judge_prompt(prompt: str, response_a: str, response_b: str) -> str:
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

def load_test_prompts(
    splits_file: str = DEFAULT_SPLITS_FILE,
    sample_size: Optional[int] = None,
    seed: int = 42
) -> List[Dict]:

    print(f"Loading test prompts from {splits_file}...")
    
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    test_data = splits['test']
    print(f"  Found {len(test_data)} test examples")
    
    # Convert to expected format
    prompts = []
    for i, example in enumerate(test_data):
        prompts.append({
            'prompt_id': example.get('example_id', f'test_{i}'),
            'prompt': example['prompt'],
            'original_chosen': example.get('chosen_response', ''),
            'original_rejected': example.get('rejected_response', ''),
        })
    
    # Sample if requested
    if sample_size and sample_size < len(prompts):
        random.seed(seed)
        prompts = random.sample(prompts, sample_size)
        print(f"  Sampled {len(prompts)} prompts")
    
    return prompts

# rollout generation
def generate_rollouts(
    client: OpenAI,
    prompt: str,
    num_rollouts: int = DEFAULT_NUM_ROLLOUTS,
    generator_model: str = DEFAULT_GENERATOR_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> List[str]:

    rollouts = []
    
    for i in range(num_rollouts):
        try:
            def make_request():
                return client.chat.completions.create(
                    model=generator_model,
                    messages=[
                        {"role": "user", "content": prompt}
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

def judge_pair(
    client: OpenAI,
    prompt: str,
    response_a: str,
    response_b: str,
    judge_model: str
) -> Optional[str]:

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

# elo updates
def update_elo(rating_a: float, rating_b: float, winner: str, k: float = DEFAULT_ELO_K) -> Tuple[float, float]:

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


def elo_to_rewards(elo_ratings: List[float]) -> List[float]:

    if len(elo_ratings) == 0:
        return []
    
    if len(elo_ratings) == 1:
        return [0.5]
    
    min_r = min(elo_ratings)
    max_r = max(elo_ratings)
    
    # Min-max normalization to [0, 1]
    spread = max_r - min_r
    if spread == 0:
        return [0.5] * len(elo_ratings)
    return [(r - min_r) / spread for r in elo_ratings]


# implementation of the tournament

def run_elo_tournament(
    client: OpenAI,
    prompt: str,
    rollouts: List[str],
    judge_model: str,
    k_factor: float = DEFAULT_ELO_K,
    verbose: bool = True
) -> Tuple[List[float], List[float], int]:

    n = len(rollouts)

    elo_ratings = [0.0] * n

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
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
    rewards = elo_to_rewards(elo_ratings)
    
    return elo_ratings, rewards, valid_comparisons

def process_single_prompt(
    client: OpenAI,
    prompt_data: Dict,
    generator_model: str,
    judge_model: str,
    num_rollouts: int,
    temperature: float,
    verbose: bool = True
) -> Optional[RolloutResult]:
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
    client: OpenAI,
    prompts: List[Dict],
    generator_model: str,
    judge_model: str,
    num_rollouts: int = DEFAULT_NUM_ROLLOUTS,
    temperature: float = DEFAULT_TEMPERATURE,
    verbose: bool = True
) -> Tuple[List[RolloutResult], List[TrainingExample]]:
 
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


def save_results(
    results: List[RolloutResult],
    training_examples: List[TrainingExample],
    output_dir: str = DEFAULT_OUTPUT_DIR
):
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    results_file = output_path / REWARDS_OUTPUT_FILE
    results_data = [asdict(r) for r in results]
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"\nSaved full results to: {results_file}")

    training_file = output_path / TRAINING_DATA_FILE
    training_data = [asdict(ex) for ex in training_examples]
    with open(training_file, 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"Saved training data to: {training_file}")
    
    return results_file, training_file


def load_results(filepath: str) -> List[Dict]:
    with open(filepath, 'r') as f:
        return json.load(f)


def main():
    # using argument parser
    parser = argparse.ArgumentParser(
        description="Generate rollouts and run ELO tournament for preference learning"
    )
    
    # Model arguments
    parser.add_argument(
        "--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
        help=f"Fine-tuned judge model ID (default: {DEFAULT_JUDGE_MODEL})"
    )
    parser.add_argument(
        "--generator-model", type=str, default=DEFAULT_GENERATOR_MODEL,
        help=f"Model for generating rollouts (default: {DEFAULT_GENERATOR_MODEL})"
    )
    
    # Data arguments
    parser.add_argument(
        "--splits-file", type=str, default=DEFAULT_SPLITS_FILE,
        help=f"Path to splits file (default: {DEFAULT_SPLITS_FILE})"
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
    
    # Output arguments
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
    
    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)
    
    # Load prompts from splits file
    prompts = load_test_prompts(
        splits_file=args.splits_file,
        sample_size=args.num_questions,
        seed=args.seed
    )
    
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
    
    print(f"\n Pipeline complete!")
    print(f"  Training examples ready for GRPO: {len(training_examples)}")
    print(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
