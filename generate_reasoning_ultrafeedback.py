import json
import random
import argparse
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from together import Together
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_OUTPUT_DIR = "data"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Output files
SPLITS_FILE = "ultrafeedback_splits.json"
REASONING_OUTPUT_FILE = "reasoning_traces.json"
SFT_OUTPUT_FILE = "ultrafeedback_preference_sft.json"

SYSTEM_MESSAGE = """You are an expert at analyzing responses to questions and instructions. Compare two responses objectively and determine which one is better and why. Consider factors like helpfulness, accuracy, relevance, clarity, and overall usefulness to the person asking the question."""

def load_ultrafeedback_dataset(sample_size: Optional[int] = None) -> List[Dict]:
    print("Loading UltraFeedback dataset...")
    ds = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")
    dataset = ds['train']
    
    if sample_size:
        dataset = dataset.select(range(min(sample_size, len(dataset))))

    data = []
    for i, example in enumerate(dataset):
        chosen_content = ""
        rejected_content = ""
        
        if len(example['chosen']) > 1:
            chosen_content = example['chosen'][1].get('content', '')
        if len(example['rejected']) > 1:
            rejected_content = example['rejected'][1].get('content', '')

        if not chosen_content or not rejected_content:
            continue
            
        data.append({
            'example_id': f"uf_{i}",
            'prompt': example['prompt'],
            'chosen_response': chosen_content,
            'rejected_response': rejected_content,
            'true_label': 1,  # 1 = chosen preferred, 0 = rejected preferred
        })
    
    print(f"Loaded {len(data)} valid examples")
    return data


def create_splits(
    data: List[Dict],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_data = shuffled[:train_end]
    val_data = shuffled[train_end:val_end]
    test_data = shuffled[val_end:]
    
    print(f"\nData splits created:")
    print(f"  Train: {len(train_data)} ({len(train_data)/n*100:.1f}%)")
    print(f"  Val:   {len(val_data)} ({len(val_data)/n*100:.1f}%)")
    print(f"  Test:  {len(test_data)} ({len(test_data)/n*100:.1f}%)")
    
    return train_data, val_data, test_data


def save_splits(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    output_dir: str = DEFAULT_OUTPUT_DIR
) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    splits_file = output_path / SPLITS_FILE
    
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'metadata': {
            'train_size': len(train_data),
            'val_size': len(val_data),
            'test_size': len(test_data),
            'total_size': len(train_data) + len(val_data) + len(test_data)
        }
    }
    
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\nSplits saved to: {splits_file}")
    return str(splits_file)


def load_splits(filepath: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    print(f"Loading splits from {filepath}...")
    with open(filepath, 'r') as f:
        splits = json.load(f)
    
    train_data = splits['train']
    val_data = splits['val']
    test_data = splits['test']
    
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    return train_data, val_data, test_data

def format_reasoning_prompt(prompt: str, response_a: str, response_b: str) -> str:
    return f"""A user asked a question and received two different responses.
Analyze which response is better and explain why.

User's Question/Instruction:
{prompt}

Response A:
{response_a}

Response B:
{response_b}

Think through this step-by-step:
1. First, I'll analyze the helpfulness and relevance of each response...
2. Next, I'll examine the clarity and completeness...
3. Then, I'll evaluate the accuracy and usefulness...
4. Finally, I'll assess the overall quality...

**ANSWER IN THE FOLLOWING FORMAT:**
ANSWER: [A/B]
REASONING: Response [A/B] is better because..."""


def generate_reasoning_traces(
    train_data: List[Dict],
    client: Together,
    model: str = DEFAULT_MODEL,
    sample_size: Optional[int] = None,
    save_interval: int = 50,
    output_file: Optional[str] = None
) -> List[Dict]:

    # Sample if needed
    if sample_size and sample_size < len(train_data):
        data_to_process = random.sample(train_data, sample_size)
    else:
        data_to_process = train_data
    
    reasoning_data = []
    agreements = 0
    total_processed = 0

    print(f"\n{'='*60}")
    print(f"GENERATING REASONING TRACES")
    print(f"{'='*60}")
    print(f"Examples to process: {len(data_to_process)}")
    print(f"Model: {model}")
    print(f"{'='*60}\n")

    for i, example in enumerate(data_to_process):
        print(f"[{i+1}/{len(data_to_process)}] Processing {example['example_id']}...")
        
        try:
            if random.random() < 0.5:
                # chosen = A, rejected = B
                response_a = example['chosen_response']
                response_b = example['rejected_response']
                correct_answer = 'A'  # A is the chosen (correct) one
            else:
                # rejected = A, chosen = B
                response_a = example['rejected_response']
                response_b = example['chosen_response']
                correct_answer = 'B'  # B is the chosen (correct) one
            
            reasoning_prompt = format_reasoning_prompt(
                example['prompt'],
                response_a,
                response_b
            )
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": reasoning_prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            reasoning_response = response.choices[0].message.content.strip()

            model_choice = None
            
            if "ANSWER: A" in reasoning_response or "ANSWER:A" in reasoning_response:
                model_choice = 'A'
            elif "ANSWER: B" in reasoning_response or "ANSWER:B" in reasoning_response:
                model_choice = 'B'
            else:
                # Fallback parsing
                if "Response A is better" in reasoning_response:
                    model_choice = 'A'
                elif "Response B is better" in reasoning_response:
                    model_choice = 'B'
                else:
                    print(f"  Could not parse answer, skipping...")
                    continue

            # Check if model agrees with ground truth
            agrees_with_truth = (model_choice == correct_answer)
            agreements += int(agrees_with_truth)
            total_processed += 1

            status = "✓ AGREES" if agrees_with_truth else "✗ DISAGREES"
            print(f"  Model: {model_choice} | Correct: {correct_answer} | {status}")

            # Only keep agreements (avoid sycophancy)
            if agrees_with_truth:
                reasoning_data.append({
                    'example_id': example['example_id'],
                    'prompt': example['prompt'],
                    'chosen_response': example['chosen_response'],
                    'rejected_response': example['rejected_response'],
                    'response_a': response_a,
                    'response_b': response_b,
                    'correct_answer': correct_answer,
                    'model_choice': model_choice,
                    'full_reasoning_response': reasoning_response,
                })
                print(f"  → Added to training data")
            else:
                print(f"  → Discarded (disagreement)")

            # Print running stats
            agreement_rate = agreements / total_processed if total_processed > 0 else 0
            print(f"  Running agreement rate: {agreement_rate:.1%} ({agreements}/{total_processed})")
            
            # Save intermediate results
            if output_file and (i + 1) % save_interval == 0:
                with open(output_file, 'w') as f:
                    json.dump(reasoning_data, f, indent=2)
                print(f"  [Saved intermediate results: {len(reasoning_data)} examples]")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"\n{'='*60}")
    print("REASONING GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total processed: {total_processed}")
    if total_processed > 0:
        print(f"Agreements: {agreements} ({agreements/total_processed:.1%})")
    print(f"Training examples created: {len(reasoning_data)}")
    print(f"{'='*60}")

    return reasoning_data

def extract_reasoning_from_response(reasoning_response: str, correct_choice: str) -> str:
    try:
        if "REASONING:" in reasoning_response:
            reasoning_part = reasoning_response.split("REASONING:", 1)[1].strip()
            
            # Clean up common prefixes
            for prefix in [
                f"Response {correct_choice} is better because",
                f"Response {correct_choice} is better",
                f"Response {correct_choice} is the better choice because",
            ]:
                if reasoning_part.lower().startswith(prefix.lower()):
                    reasoning_part = reasoning_part[len(prefix):].strip()
            
            # Remove leading punctuation
            reasoning_part = reasoning_part.lstrip(":.,- ")
            
            if reasoning_part:
                return reasoning_part

        if "better because" in reasoning_response.lower():
            parts = reasoning_response.lower().split("better because", 1)
            if len(parts) > 1:
                return parts[1].strip().lstrip(":.,- ")
        
        return "it provides a more helpful, accurate, and relevant response to the question asked."
        
    except Exception:
        return "it better addresses the user's needs and provides more valuable information."


def generate_sft_dataset(
    reasoning_data: List[Dict],
    output_file: str
) -> List[Dict]:

    sft_dataset = []

    print(f"\n{'='*60}")
    print(f"GENERATING SFT DATASET")
    print(f"{'='*60}")
    print(f"Input examples: {len(reasoning_data)}")

    for i, example in enumerate(reasoning_data):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i+1}/{len(reasoning_data)}...")

        # Extract data
        prompt = example['prompt']
        chosen_response = example['chosen_response']
        rejected_response = example['rejected_response']
        reasoning_response = example['full_reasoning_response']

        # Determine correct choice based on randomized positions
        correct_choice = example['correct_answer']
        
        # Extract reasoning
        reasoning_text = extract_reasoning_from_response(reasoning_response, correct_choice)

        flip_positions = random.choice([True, False])

        if flip_positions:
            display_response_a = rejected_response
            display_response_b = chosen_response
            display_correct_choice = 'B'  # chosen is now B
        else:
            display_response_a = chosen_response
            display_response_b = rejected_response
            display_correct_choice = 'A'  # chosen is A

        user_prompt = f"""Which response is better? Analyze the differences between these two responses.

Original Post:
{prompt}

Response A:
{display_response_a}

Response B:
{display_response_b}

Think through this step-by-step:
1. First, I'll analyze the helpfulness and relevance...
2. Next, I'll examine the clarity and completeness...
3. Then, I'll evaluate the accuracy and usefulness...
4. Finally, I'll assess the overall quality...

**ANSWER IN THE FOLLOWING FORMAT:**
Response [A/B] is better because..."""

        # Create the assistant response
        assistant_response = f"Response {display_correct_choice} is better because {reasoning_text}"

        # Create SFT training example
        sft_example = {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
                {
                    "role": "assistant",
                    "content": assistant_response
                }
            ],
            "metadata": {
                "example_id": example['example_id'],
                "flipped": flip_positions,
                "correct_choice": display_correct_choice
            }
        }

        sft_dataset.append(sft_example)

    # Save the dataset
    with open(output_file, 'w') as f:
        json.dump(sft_dataset, f, indent=2)

    # Print statistics
    flipped_count = sum(1 for ex in sft_dataset if ex['metadata']['flipped'])
    a_count = sum(1 for ex in sft_dataset if ex['metadata']['correct_choice'] == 'A')
    
    print(f"\n{'='*60}")
    print("SFT DATASET GENERATED")
    print(f"{'='*60}")
    print(f"Total examples: {len(sft_dataset)}")
    print(f"Position flips: {flipped_count}/{len(sft_dataset)} ({flipped_count/len(sft_dataset)*100:.1f}%)")
    print(f"Correct=A: {a_count}, Correct=B: {len(sft_dataset)-a_count}")
    print(f"Saved to: {output_file}")
    print(f"{'='*60}")

    return sft_dataset


def preview_sft_examples(sft_dataset: List[Dict], num_examples: int = 2):
    print(f"\n{'='*80}")
    print(f"PREVIEW OF SFT DATASET ({num_examples} examples)")
    print(f"{'='*80}")

    for i in range(min(num_examples, len(sft_dataset))):
        example = sft_dataset[i]
        print(f"\n--- EXAMPLE {i+1} ---")
        print(f"Example ID: {example['metadata']['example_id']}")
        print(f"Flipped: {example['metadata']['flipped']}")
        print(f"Correct: {example['metadata']['correct_choice']}")
        print(f"\nUSER PROMPT (truncated):")
        print(example['messages'][1]['content'][:400] + "...")
        print(f"\nASSISTANT RESPONSE:")
        print(example['messages'][2]['content'][:300] + "...")
        print("-" * 80)

# main
def main():
    parser = argparse.ArgumentParser(
        description="Generate reasoning traces and SFT dataset for UltraFeedback"
    )
    
    # Data arguments
    parser.add_argument(
        "--full-dataset-size", type=int, default=30000,
        help="Size of full dataset to load before splitting (default: 30000)"
    )
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of TRAIN examples to process for reasoning (default: {DEFAULT_SAMPLE_SIZE})"
    )
    
    # Model arguments
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Together AI model for reasoning (default: {DEFAULT_MODEL})"
    )
    
    # I/O arguments
    parser.add_argument(
        "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--splits-file", type=str, default=None,
        help="Load existing splits file instead of creating new ones"
    )
    parser.add_argument(
        "--reasoning-file", type=str, default=None,
        help="Load existing reasoning traces instead of generating"
    )
    
    # Mode arguments
    parser.add_argument(
        "--splits-only", action="store_true",
        help="Only create and save splits, don't generate reasoning"
    )
    parser.add_argument(
        "--preview", action="store_true",
        help="Preview SFT examples after generation"
    )
    
    # Other arguments
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--save-interval", type=int, default=50,
        help="Save intermediate results every N examples (default: 50)"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output paths
    splits_path = output_dir / SPLITS_FILE
    reasoning_path = output_dir / REASONING_OUTPUT_FILE
    sft_path = output_dir / SFT_OUTPUT_FILE

    if args.splits_file:
        # Load existing splits
        train_data, val_data, test_data = load_splits(args.splits_file)
    else:
        # Create new splits
        print("\n" + "="*60)
        print("STEP 1: LOADING DATA AND CREATING SPLITS")
        print("="*60)
        
        full_data = load_ultrafeedback_dataset(sample_size=args.full_dataset_size)
        train_data, val_data, test_data = create_splits(
            full_data,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            seed=args.seed
        )
        save_splits(train_data, val_data, test_data, args.output_dir)
    
    if args.splits_only:
        print("\n✓ Splits created. Exiting (--splits-only mode)")
        return

    if args.reasoning_file:
        # Load existing reasoning traces
        print(f"\nLoading existing reasoning traces from {args.reasoning_file}...")
        with open(args.reasoning_file, 'r') as f:
            reasoning_data = json.load(f)
        print(f"Loaded {len(reasoning_data)} reasoning traces")
    else:
        # Generate new reasoning traces
        print("\n" + "="*60)
        print("STEP 2: GENERATING REASONING TRACES")
        print("="*60)
        
        # Initialize Together AI client
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")
        client = Together(api_key=api_key)
        
        reasoning_data = generate_reasoning_traces(
            train_data=train_data,
            client=client,
            model=args.model,
            sample_size=args.sample_size,
            save_interval=args.save_interval,
            output_file=str(reasoning_path)
        )
        
        # Save final reasoning traces
        with open(reasoning_path, 'w') as f:
            json.dump(reasoning_data, f, indent=2)
        print(f"\nReasoning traces saved to: {reasoning_path}")

    print("\n" + "="*60)
    print("STEP 3: GENERATING SFT DATASET")
    print("="*60)
    
    sft_dataset = generate_sft_dataset(reasoning_data, str(sft_path))
    
    # Optional preview
    if args.preview:
        preview_sft_examples(sft_dataset)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  Splits:    {splits_path}")
    print(f"  Reasoning: {reasoning_path}")
    print(f"  SFT Data:  {sft_path}")
    print(f"\nNext step: Run finetune_ultrafeedback.py to train the expert model")
    print(f"\n  python finetune_ultrafeedback.py train --input {sft_path}")


if __name__ == "__main__":
    main()
