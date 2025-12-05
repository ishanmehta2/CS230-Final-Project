from datasets import load_dataset
import json
from typing import Dict, List
import random
import os
import boto3
from dotenv import load_dotenv

load_dotenv()


def format_baseline_prompt(instruction, response_a, response_b):
    prompt = f"""Given the following instruction and two responses, determine which response is better.

INSTRUCTION: {instruction}

RESPONSE A: {response_a}

RESPONSE B: {response_b}

Which response is better? Respond with only "A" or "B"."""

    return prompt


def prepare_baseline_data(dataset, split='train'):
    baseline_data = []
    for example in dataset[split]:
        # Randomly assign chosen/rejected to A/B to avoid position bias
        if random.random() < 0.5:
            response_a = example['chosen']
            response_b = example['rejected']
            true_label = 'A'  # A is better (chosen)
        else:
            response_a = example['rejected']
            response_b = example['chosen']
            true_label = 'B'  # B is better (chosen)
        
        formatted_example = {
            'instruction': example['prompt'],
            'response_A': response_a,
            'response_B': response_b,
            'true_label': true_label,
            'chosen_avg_rating': example['chosen-rating'],
            'rejected_avg_rating': example['rejected-rating'],
            'source': example['source'],
            'prompt': format_baseline_prompt(
                example['prompt'],
                response_a,
                response_b
            )
        }
        baseline_data.append(formatted_example)
    return baseline_data


def run_baseline_evaluation(baseline_data, bedrock_client, sample_size=None):
    if sample_size:
        baseline_data = random.sample(baseline_data, sample_size)

    correct = 0
    total = 0
    results = []

    for idx, example in enumerate(baseline_data):
        try:
            # Using Meta Llama 3.1 70B Instruct on Bedrock
            model_id = "meta.llama3-70b-instruct-v1:0"
            
            # llama formatting for bedrock
            formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{example['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            request_body = json.dumps({
                "prompt": formatted_prompt,
                "max_gen_len": 50,
                "temperature": 0.1,
                "top_p": 0.9
            })
            
            response = bedrock_client.invoke_model(
                modelId=model_id,
                body=request_body
            )
            
            response_body = json.loads(response['body'].read())
            model_answer = response_body.get('generation', '').strip().upper()
            
            # Try to extract A or B from the response
            if 'A' in model_answer and 'B' not in model_answer:
                model_answer = 'A'
            elif 'B' in model_answer and 'A' not in model_answer:
                model_answer = 'B'
            elif model_answer.startswith('A'):
                model_answer = 'A'
            elif model_answer.startswith('B'):
                model_answer = 'B'
            else:
                # Try finding A or B in the first few characters
                first_chars = model_answer[:10]
                if 'A' in first_chars:
                    model_answer = 'A'
                elif 'B' in first_chars:
                    model_answer = 'B'

            if model_answer not in ['A', 'B']:
                continue

            # Check if correct
            is_correct = (model_answer == example['true_label'])
            correct += is_correct
            total += 1

            results.append({
                'instruction': example['instruction'][:100] + '...',
                'model_prediction': model_answer,
                'true_label': example['true_label'],
                'correct': is_correct,
                'model_response': model_answer,
                'chosen_rating': example['chosen_avg_rating'],
                'rejected_rating': example['rejected_avg_rating']
            })

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            continue

    accuracy = correct / total if total > 0 else 0
    print(f"Baseline Accuracy: {accuracy:.3f} ({correct}/{total})")

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'results': results
    }


def main():
    # Initialize AWS Bedrock client
    bedrock_client = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1',
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN")
    )
    
    # Load dataset
    ds = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")
    print(f"Dataset size - Train: {len(ds['train'])}")

    # Prepare data
    train_baseline = prepare_baseline_data(ds, 'train')

    # Run evaluation
    aggregate_accuracy = 0.0
    num_runs = 10
    
    for i in range(num_runs):
        print(f"\nRun {i+1}/{num_runs}")
        output = run_baseline_evaluation(train_baseline, bedrock_client, sample_size=100)
        aggregate_accuracy += output['accuracy']
    
    print(f"\nAverage accuracy is {aggregate_accuracy/num_runs:.3f}")


if __name__ == "__main__":
    main()
