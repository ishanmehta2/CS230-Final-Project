"""
Fine-tune and Evaluate Expert Judge Model on Together AI (UltraFeedback)

This script:
1. Converts SFT JSON to Together AI JSONL format
2. Uploads and launches fine-tuning job
3. Monitors job status
4. Evaluates fine-tuned model on held-out test set
5. Compares to baseline model

Usage:
    # Full training pipeline
    python finetune_ultrafeedback.py train --input data/ultrafeedback_preference_sft.json --wait
    
    # Check job status
    python finetune_ultrafeedback.py status <job_id>
    
    # Evaluate fine-tuned model
    python finetune_ultrafeedback.py evaluate <model_id> --compare-baseline
    
    # Run baseline only
    python finetune_ultrafeedback.py baseline --num-examples 200
"""

import json
import argparse
import time
import random
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from together import Together
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

# Together AI fine-tunable models
# See: https://docs.together.ai/docs/fine-tuning-models
DEFAULT_BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"

# For baseline comparison
BASELINE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

# Training defaults
DEFAULT_EPOCHS = 2
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_BATCH_SIZE = 8

# File paths
DEFAULT_INPUT_FILE = "data/ultrafeedback_preference_sft.json"
DEFAULT_JSONL_FILE = "data/together_format.jsonl"
DEFAULT_SPLITS_FILE = "data/ultrafeedback_splits.json"

# Prompt template (must match training format exactly)
SYSTEM_MESSAGE = """You are an expert at analyzing responses to questions and instructions. Compare two responses objectively and determine which one is better and why. Consider factors like helpfulness, accuracy, relevance, clarity, and overall usefulness to the person asking the question."""


# =============================================================================
# FORMAT CONVERSION
# =============================================================================

def convert_to_together_jsonl(input_file: str, output_file: str) -> int:
    """
    Convert SFT JSON to Together AI JSONL format.
    
    Together expects JSONL with "messages" array containing role/content objects.
    """
    print(f"\nConverting {input_file} to Together JSONL format...")
    
    with open(input_file, 'r') as f:
        sft_data = json.load(f)

    with open(output_file, 'w') as f:
        for example in sft_data:
            # Together format uses the same messages structure
            together_example = {
                "messages": example["messages"]
            }
            json.dump(together_example, f)
            f.write('\n')

    print(f"  Converted {len(sft_data)} examples")
    print(f"  Saved to: {output_file}")
    
    return len(sft_data)


# =============================================================================
# FINE-TUNING OPERATIONS
# =============================================================================

def upload_training_file(client: Together, jsonl_file: str) -> str:
    """Upload training file to Together AI."""
    print(f"\nUploading {jsonl_file} to Together AI...")
    
    response = client.files.upload(
        file=jsonl_file,
        purpose='fine-tune'
    )
    
    file_id = response.id
    print(f"  File ID: {file_id}")
    
    return file_id


def launch_finetuning(
    client: Together,
    file_id: str,
    base_model: str = DEFAULT_BASE_MODEL,
    n_epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    suffix: Optional[str] = None
) -> str:
    """Launch Together AI fine-tuning job."""
    
    print(f"\n{'='*50}")
    print("LAUNCHING FINE-TUNING JOB")
    print(f"{'='*50}")
    print(f"Base model: {base_model}")
    print(f"Epochs: {n_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Suffix: {suffix or 'preference-expert'}")
    
    job = client.fine_tuning.create(
        training_file=file_id,
        model=base_model,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        suffix=suffix or "preference-expert"
    )
    
    job_id = job.id
    print(f"\nJob ID: {job_id}")
    print(f"Status: {job.status}")
    
    return job_id


def check_status(client: Together, job_id: str) -> Tuple[Optional[str], str]:
    """
    Check fine-tuning job status.
    
    Returns:
        Tuple of (model_id or None, status_string)
    """
    job = client.fine_tuning.retrieve(id=job_id)
    
    print(f"Status: {job.status}")
    
    if job.status == "completed":
        model_name = getattr(job, 'output_name', None) or getattr(job, 'fine_tuned_model', None)
        print(f"✓ Model ready: {model_name}")
        return model_name, "completed"
        
    elif job.status == "failed":
        error_msg = getattr(job, 'error', "Unknown error")
        print(f"✗ Failed: {error_msg}")
        return None, "failed"
        
    elif job.status == "cancelled":
        print(f"✗ Cancelled")
        return None, "cancelled"
    
    # Show progress if available
    if hasattr(job, 'training_steps') and job.training_steps:
        print(f"Training steps: {job.training_steps}")
    
    return None, job.status


def wait_for_completion(client: Together, job_id: str, check_interval: int = 60) -> Optional[str]:
    """Wait for fine-tuning job to complete."""
    
    print(f"\nWaiting for fine-tuning to complete...")
    print(f"Checking every {check_interval} seconds")
    print("-" * 40)
    
    iteration = 0
    while True:
        iteration += 1
        print(f"\n[Check #{iteration}]")
        
        model_id, status = check_status(client, job_id)
        
        if status == "completed":
            return model_id
        elif status in ["failed", "cancelled"]:
            return None
        
        print(f"Waiting {check_interval}s...")
        time.sleep(check_interval)


def list_jobs(client: Together, limit: int = 10) -> List:
    """List recent fine-tuning jobs."""
    
    print(f"\nRecent fine-tuning jobs (limit {limit}):")
    print("-" * 60)
    
    jobs = client.fine_tuning.list()
    
    for i, job in enumerate(jobs.data[:limit]):
        model_str = ""
        if hasattr(job, 'output_name') and job.output_name:
            model_str = f" → {job.output_name}"
        print(f"  {job.id}: {job.status} ({job.model}){model_str}")
    
    return jobs.data[:limit]


def cancel_job(client: Together, job_id: str):
    """Cancel a fine-tuning job."""
    print(f"Cancelling job {job_id}...")
    client.fine_tuning.cancel(id=job_id)
    print("  Job cancelled")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_test_data(splits_file: str = DEFAULT_SPLITS_FILE) -> List[Dict]:
    """Load test data from saved splits."""
    
    print(f"\nLoading test data from {splits_file}...")
    
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    test_data = splits['test']
    print(f"  Loaded {len(test_data)} test examples")
    
    return test_data


def load_val_data(splits_file: str = DEFAULT_SPLITS_FILE) -> List[Dict]:
    """Load validation data from saved splits."""
    
    print(f"\nLoading validation data from {splits_file}...")
    
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    val_data = splits['val']
    print(f"  Loaded {len(val_data)} validation examples")
    
    return val_data


# =============================================================================
# EVALUATION PROMPTS
# =============================================================================

def format_baseline_prompt(prompt: str, response_a: str, response_b: str) -> str:
    """Format simple prompt for baseline evaluation."""
    return f"""Given the following question and two responses, determine which response is better.

QUESTION: {prompt}

RESPONSE A: {response_a}

RESPONSE B: {response_b}

Which response is better? Respond with only "A" or "B"."""


def format_expert_prompt(prompt: str, response_a: str, response_b: str) -> str:
    """
    Format prompt for expert model evaluation.
    MUST match the exact format used during fine-tuning.
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
# EVALUATION FUNCTIONS
# =============================================================================

def run_baseline_evaluation(
    client: Together,
    test_data: List[Dict],
    model: str = BASELINE_MODEL,
    sample_size: int = 100,
    seed: int = 42
) -> Dict:
    """
    Run baseline evaluation using simple A/B prompt.
    """
    random.seed(seed)
    
    print(f"\n{'='*50}")
    print("BASELINE EVALUATION")
    print(f"{'='*50}")
    print(f"Model: {model}")
    print(f"Examples: {min(sample_size, len(test_data))}")
    print("-" * 50)
    
    if sample_size < len(test_data):
        test_data = random.sample(test_data, sample_size)

    correct = 0
    total = 0
    results = []

    for i, example in enumerate(test_data):
        if (i + 1) % 20 == 0:
            print(f"  Processing {i+1}/{len(test_data)}... (acc: {correct/total:.3f})" if total > 0 else f"  Processing {i+1}/{len(test_data)}...")
        
        try:
            # Randomize position to avoid bias in evaluation
            if random.random() < 0.5:
                response_a = example['chosen_response']
                response_b = example['rejected_response']
                correct_answer = 'A'
            else:
                response_a = example['rejected_response']
                response_b = example['chosen_response']
                correct_answer = 'B'
            
            prompt = format_baseline_prompt(
                example['prompt'],
                response_a,
                response_b
            )
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )

            model_answer = response.choices[0].message.content.strip().upper()

            # Parse answer
            if 'A' in model_answer and 'B' not in model_answer:
                model_choice = 'A'
            elif 'B' in model_answer and 'A' not in model_answer:
                model_choice = 'B'
            elif model_answer.startswith('A'):
                model_choice = 'A'
            elif model_answer.startswith('B'):
                model_choice = 'B'
            else:
                print(f"    Could not parse: {model_answer}")
                continue

            is_correct = (model_choice == correct_answer)
            correct += int(is_correct)
            total += 1

            results.append({
                'example_id': example['example_id'],
                'model_choice': model_choice,
                'correct_answer': correct_answer,
                'correct': is_correct,
            })

        except Exception as e:
            print(f"    Error: {e}")
            continue

    accuracy = correct / total if total > 0 else 0
    
    print(f"\n{'='*50}")
    print("BASELINE RESULTS")
    print(f"{'='*50}")
    print(f"Model: {model}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"{'='*50}")

    return {
        'model': model,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results
    }


def evaluate_finetuned_model(
    client: Together,
    model_id: str,
    test_data: List[Dict],
    sample_size: int = 100,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Evaluate fine-tuned model using the exact training prompt format.
    """
    random.seed(seed)
    
    print(f"\n{'='*50}")
    print("FINE-TUNED MODEL EVALUATION")
    print(f"{'='*50}")
    print(f"Model: {model_id}")
    print(f"Examples: {min(sample_size, len(test_data))}")
    print("-" * 50)

    if sample_size < len(test_data):
        test_data = random.sample(test_data, sample_size)

    correct = 0
    total = 0
    results = []

    for i, example in enumerate(test_data):
        if verbose and (i + 1) % 10 == 0:
            acc_str = f" (acc: {correct/total:.3f})" if total > 0 else ""
            print(f"  Processing {i+1}/{len(test_data)}...{acc_str}")

        try:
            # Randomize position
            if random.random() < 0.5:
                response_a = example['chosen_response']
                response_b = example['rejected_response']
                correct_answer = 'A'
            else:
                response_a = example['rejected_response']
                response_b = example['chosen_response']
                correct_answer = 'B'
            
            prompt = format_expert_prompt(
                example['prompt'],
                response_a,
                response_b
            )

            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0
            )

            full_response = response.choices[0].message.content.strip()

            # Parse the choice
            model_choice = None
            if "Response A is better" in full_response:
                model_choice = "A"
            elif "Response B is better" in full_response:
                model_choice = "B"
            else:
                # Fallback parsing
                if full_response.startswith("Response A") or "prefer A" in full_response.lower():
                    model_choice = "A"
                elif full_response.startswith("Response B") or "prefer B" in full_response.lower():
                    model_choice = "B"
                else:
                    if verbose:
                        print(f"    Could not parse response")
                    continue

            is_correct = (model_choice == correct_answer)
            correct += int(is_correct)
            total += 1

            results.append({
                'example_id': example['example_id'],
                'model_choice': model_choice,
                'correct_answer': correct_answer,
                'correct': is_correct,
                'reasoning': full_response[:200]
            })

        except Exception as e:
            print(f"    Error: {e}")
            continue

    accuracy = correct / total if total > 0 else 0
    
    print(f"\n{'='*50}")
    print("FINE-TUNED MODEL RESULTS")
    print(f"{'='*50}")
    print(f"Model: {model_id}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"{'='*50}")

    return {
        'model_id': model_id,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results
    }


def compare_models(baseline_results: Dict, finetuned_results: Dict):
    """Compare baseline and fine-tuned model results."""
    
    baseline_acc = baseline_results['accuracy']
    finetuned_acc = finetuned_results['accuracy']
    improvement = finetuned_acc - baseline_acc
    relative_improvement = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0
    
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"\nBaseline ({baseline_results['model']}):")
    print(f"  Accuracy: {baseline_acc:.4f} ({baseline_results['correct']}/{baseline_results['total']})")
    
    print(f"\nFine-tuned ({finetuned_results['model_id']}):")
    print(f"  Accuracy: {finetuned_acc:.4f} ({finetuned_results['correct']}/{finetuned_results['total']})")
    
    print(f"\nImprovement:")
    print(f"  Absolute: {improvement:+.4f}")
    print(f"  Relative: {relative_improvement:+.1f}%")
    
    if improvement > 0:
        print(f"\n✓ Fine-tuned model outperforms baseline!")
    elif improvement < 0:
        print(f"\n✗ Baseline outperforms fine-tuned model")
    else:
        print(f"\n= Models perform equally")
    
    print(f"{'='*60}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune and evaluate preference expert model on Together AI"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # -------------------------------------------------------------------------
    # CONVERT command
    # -------------------------------------------------------------------------
    convert_parser = subparsers.add_parser("convert", help="Convert SFT JSON to JSONL")
    convert_parser.add_argument(
        "--input", type=str, default=DEFAULT_INPUT_FILE,
        help=f"Input SFT JSON file (default: {DEFAULT_INPUT_FILE})"
    )
    convert_parser.add_argument(
        "--output", type=str, default=DEFAULT_JSONL_FILE,
        help=f"Output JSONL file (default: {DEFAULT_JSONL_FILE})"
    )
    
    # -------------------------------------------------------------------------
    # TRAIN command
    # -------------------------------------------------------------------------
    train_parser = subparsers.add_parser("train", help="Launch fine-tuning")
    train_parser.add_argument(
        "--input", type=str, default=DEFAULT_INPUT_FILE,
        help=f"Input SFT JSON file (default: {DEFAULT_INPUT_FILE})"
    )
    train_parser.add_argument(
        "--jsonl", type=str, default=DEFAULT_JSONL_FILE,
        help=f"JSONL file path (default: {DEFAULT_JSONL_FILE})"
    )
    train_parser.add_argument(
        "--model", type=str, default=DEFAULT_BASE_MODEL,
        help=f"Base model to fine-tune (default: {DEFAULT_BASE_MODEL})"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Number of epochs (default: {DEFAULT_EPOCHS})"
    )
    train_parser.add_argument(
        "--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
        help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})"
    )
    train_parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})"
    )
    train_parser.add_argument(
        "--suffix", type=str, default=None,
        help="Model name suffix"
    )
    train_parser.add_argument(
        "--wait", action="store_true",
        help="Wait for fine-tuning to complete"
    )
    train_parser.add_argument(
        "--skip-convert", action="store_true",
        help="Skip conversion (use existing JSONL)"
    )
    
    # -------------------------------------------------------------------------
    # STATUS command
    # -------------------------------------------------------------------------
    status_parser = subparsers.add_parser("status", help="Check job status")
    status_parser.add_argument("job_id", type=str, help="Fine-tuning job ID")
    status_parser.add_argument("--wait", action="store_true", help="Wait for completion")
    
    # -------------------------------------------------------------------------
    # LIST command
    # -------------------------------------------------------------------------
    list_parser = subparsers.add_parser("list", help="List fine-tuning jobs")
    list_parser.add_argument(
        "--limit", type=int, default=10,
        help="Number of jobs to list (default: 10)"
    )
    
    # -------------------------------------------------------------------------
    # CANCEL command
    # -------------------------------------------------------------------------
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a job")
    cancel_parser.add_argument("job_id", type=str, help="Job ID to cancel")
    
    # -------------------------------------------------------------------------
    # EVALUATE command
    # -------------------------------------------------------------------------
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate fine-tuned model")
    eval_parser.add_argument("model_id", type=str, help="Fine-tuned model ID")
    eval_parser.add_argument(
        "--splits-file", type=str, default=DEFAULT_SPLITS_FILE,
        help=f"Splits file with test data (default: {DEFAULT_SPLITS_FILE})"
    )
    eval_parser.add_argument(
        "--num-examples", type=int, default=200,
        help="Number of test examples (default: 200)"
    )
    eval_parser.add_argument(
        "--compare-baseline", action="store_true",
        help="Also run baseline for comparison"
    )
    eval_parser.add_argument(
        "--baseline-model", type=str, default=BASELINE_MODEL,
        help=f"Baseline model (default: {BASELINE_MODEL})"
    )
    eval_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    eval_parser.add_argument(
        "--use-val", action="store_true",
        help="Use validation set instead of test set"
    )
    
    # -------------------------------------------------------------------------
    # BASELINE command
    # -------------------------------------------------------------------------
    baseline_parser = subparsers.add_parser("baseline", help="Run baseline evaluation only")
    baseline_parser.add_argument(
        "--model", type=str, default=BASELINE_MODEL,
        help=f"Model to evaluate (default: {BASELINE_MODEL})"
    )
    baseline_parser.add_argument(
        "--splits-file", type=str, default=DEFAULT_SPLITS_FILE,
        help=f"Splits file (default: {DEFAULT_SPLITS_FILE})"
    )
    baseline_parser.add_argument(
        "--num-examples", type=int, default=200,
        help="Number of examples (default: 200)"
    )
    baseline_parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    baseline_parser.add_argument(
        "--use-val", action="store_true",
        help="Use validation set instead of test set"
    )
    
    args = parser.parse_args()
    
    # Initialize Together AI client
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable not set")
    client = Together(api_key=api_key)
    
    # =========================================================================
    # COMMAND HANDLERS
    # =========================================================================
    
    if args.command == "convert":
        convert_to_together_jsonl(args.input, args.output)
        
    elif args.command == "train":
        # Step 1: Convert to JSONL
        if not args.skip_convert:
            convert_to_together_jsonl(args.input, args.jsonl)
        
        # Step 2: Upload file
        file_id = upload_training_file(client, args.jsonl)
        
        # Step 3: Launch fine-tuning
        job_id = launch_finetuning(
            client=client,
            file_id=file_id,
            base_model=args.model,
            n_epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            suffix=args.suffix
        )
        
        # Step 4: Optionally wait
        if args.wait:
            model_id = wait_for_completion(client, job_id)
            if model_id:
                print(f"\n✓ Fine-tuning complete!")
                print(f"  Model ID: {model_id}")
                print(f"\n  To evaluate:")
                print(f"    python finetune_ultrafeedback.py evaluate {model_id} --compare-baseline")
            else:
                print(f"\n✗ Fine-tuning failed or was cancelled")
        else:
            print(f"\n✓ Fine-tuning job launched!")
            print(f"  Job ID: {job_id}")
            print(f"\n  To check status:")
            print(f"    python finetune_ultrafeedback.py status {job_id}")
            print(f"\n  To wait for completion:")
            print(f"    python finetune_ultrafeedback.py status {job_id} --wait")
    
    elif args.command == "status":
        if args.wait:
            model_id = wait_for_completion(client, args.job_id)
            if model_id:
                print(f"\n✓ Model ready: {model_id}")
        else:
            check_status(client, args.job_id)
    
    elif args.command == "list":
        list_jobs(client, args.limit)
    
    elif args.command == "cancel":
        cancel_job(client, args.job_id)
    
    elif args.command == "evaluate":
        # Load test or val data
        if args.use_val:
            test_data = load_val_data(args.splits_file)
        else:
            test_data = load_test_data(args.splits_file)
        
        # Evaluate fine-tuned model
        finetuned_results = evaluate_finetuned_model(
            client=client,
            model_id=args.model_id,
            test_data=test_data,
            sample_size=args.num_examples,
            seed=args.seed
        )
        
        # Optionally compare to baseline
        if args.compare_baseline:
            baseline_results = run_baseline_evaluation(
                client=client,
                test_data=test_data,
                model=args.baseline_model,
                sample_size=args.num_examples,
                seed=args.seed
            )
            compare_models(baseline_results, finetuned_results)
        
        # Save results
        results_file = f"eval_results_{args.model_id.split('/')[-1]}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'finetuned': finetuned_results,
                'baseline': baseline_results if args.compare_baseline else None
            }, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    elif args.command == "baseline":
        # Load test or val data
        if args.use_val:
            test_data = load_val_data(args.splits_file)
        else:
            test_data = load_test_data(args.splits_file)
        
        # Run baseline
        results = run_baseline_evaluation(
            client=client,
            test_data=test_data,
            model=args.model,
            sample_size=args.num_examples,
            seed=args.seed
        )
        
        # Save results
        results_file = f"baseline_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()