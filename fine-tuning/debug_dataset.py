import torch
from transformers import AutoTokenizer
from data_prep import prepare_data_splits, create_dataloaders
from collections import Counter


def debug_dataset():
    print("="*80)
    print("DATASET DEBUGGING SCRIPT")
    print("="*80)
    
    # Load tokenizer
    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # Prepare data
    print("\n=== Loading Data ===")
    train_split, val_split, test_split = prepare_data_splits(subsample_size=None)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_split, val_split, test_split,
        tokenizer,
        batch_size=16,
        max_length=256
    )
    
    # Check 1: Source distribution
    print("\n" + "="*80)
    print("CHECK 1: Source Distribution")
    print("="*80)
    
    train_sources = [train_split.dataset[i]['source'] for i in train_split.indices]
    val_sources = [val_split.dataset[i]['source'] for i in val_split.indices]
    test_sources = [test_split.dataset[i]['source'] for i in test_split.indices]
    
    print("\nTrain source distribution:")
    for source, count in Counter(train_sources).most_common():
        print(f"  {source}: {count} ({count/len(train_sources)*100:.1f}%)")
    
    print("\nVal source distribution:")
    for source, count in Counter(val_sources).most_common():
        print(f"  {source}: {count} ({count/len(val_sources)*100:.1f}%)")
    
    print("\nTest source distribution:")
    for source, count in Counter(test_sources).most_common():
        print(f"  {source}: {count} ({count/len(test_sources)*100:.1f}%)")
    
    # Check 2: Rating consistency
    print("\n" + "="*80)
    print("CHECK 2: Rating Consistency (Is 'chosen' actually better?)")
    print("="*80)
    
    batch = next(iter(train_loader))
    
    print("\nFirst 20 examples from training set:")
    for i in range(min(20, len(batch['chosen_rating']))):
        chosen_rating = batch['chosen_rating'][i].item()
        rejected_rating = batch['rejected_rating'][i].item()
        is_correct = chosen_rating > rejected_rating
        
        print(f"Example {i+1}: Chosen={chosen_rating:.2f}, Rejected={rejected_rating:.2f} | "
              f"Chosen > Rejected? {is_correct} {'✓' if is_correct else '✗'}")
    
    # Check 3: Overall label accuracy
    print("\n" + "="*80)
    print("CHECK 3: Overall Label Quality")
    print("="*80)
    
    def check_label_quality(loader, split_name):
        correct_labels = 0
        total = 0
        rating_diffs = []
        
        for batch in loader:
            chosen_ratings = batch['chosen_rating']
            rejected_ratings = batch['rejected_rating']
            
            correct_labels += (chosen_ratings > rejected_ratings).sum().item()
            total += len(chosen_ratings)
            
            diffs = (chosen_ratings - rejected_ratings).tolist()
            rating_diffs.extend(diffs)
        
        accuracy = correct_labels / total
        avg_diff = sum(rating_diffs) / len(rating_diffs)
        
        print(f"\n{split_name} set:")
        print(f"  Total examples: {total}")
        print(f"  Correct labels (chosen > rejected): {correct_labels} ({accuracy*100:.2f}%)")
        print(f"  Incorrect labels (chosen < rejected): {total - correct_labels} ({(1-accuracy)*100:.2f}%)")
        print(f"  Average rating difference (chosen - rejected): {avg_diff:.3f}")
        
        return accuracy, rating_diffs
    
    train_acc, train_diffs = check_label_quality(train_loader, "Train")
    val_acc, val_diffs = check_label_quality(val_loader, "Val")
    test_acc, test_diffs = check_label_quality(test_loader, "Test")
    
    # Check 4: Sample actual text
    print("\n" + "="*80)
    print("CHECK 4: Sample Text Examples")
    print("="*80)
    
    batch = next(iter(train_loader))
    
    print("\nExample 1 (should have chosen > rejected):")
    print(f"Chosen rating: {batch['chosen_rating'][0].item():.2f}")
    print(f"Rejected rating: {batch['rejected_rating'][0].item():.2f}")
    print("\nChosen text (first 300 chars):")
    chosen_text = tokenizer.decode(batch['chosen_input_ids'][0], skip_special_tokens=True)
    print(chosen_text[:300] + "...")
    print("\nRejected text (first 300 chars):")
    rejected_text = tokenizer.decode(batch['rejected_input_ids'][0], skip_special_tokens=True)
    print(rejected_text[:300] + "...")
    
    # Check 5: Find examples where labels are flipped
    print("\n" + "="*80)
    print("CHECK 5: Examples with Flipped Labels (rejected > chosen)")
    print("="*80)
    
    flipped_count = 0
    for batch in train_loader:
        flipped_mask = batch['rejected_rating'] > batch['chosen_rating']
        flipped_count += flipped_mask.sum().item()
        
        if flipped_mask.any():
            # Show first flipped example
            idx = flipped_mask.nonzero()[0].item()
            print(f"\nFlipped example found:")
            print(f"  Chosen rating: {batch['chosen_rating'][idx].item():.2f}")
            print(f"  Rejected rating: {batch['rejected_rating'][idx].item():.2f}")
            print(f"  Source: {batch['source'][idx]}")
            break
    
    print(f"\nTotal flipped examples in train set: {flipped_count}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n1. Dataset sizes:")
    print(f"   Train: {len(train_split)}, Val: {len(val_split)}, Test: {len(test_split)}")
    
    print(f"\n2. Label quality:")
    print(f"   Train: {train_acc*100:.2f}% correct labels")
    print(f"   Val:   {val_acc*100:.2f}% correct labels")
    print(f"   Test:  {test_acc*100:.2f}% correct labels")
    
    if train_acc < 0.95:
        print(f"\n⚠️  WARNING: {(1-train_acc)*100:.1f}% of labels are flipped!")
        print("   This explains why validation accuracy is so bad.")
        print("   We need to fix the evaluation logic to handle this.")
    
    print("\n3. Recommendations:")
    if train_acc < 0.95:
        print("   - Fix evaluation: use ratings to determine true label, not field names")
        print("   - Filter out examples where chosen_rating <= rejected_rating")
    else:
        print("   - Labels look good, problem might be elsewhere")
        print("   - Consider trying larger model (bert-base-uncased)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    debug_dataset()