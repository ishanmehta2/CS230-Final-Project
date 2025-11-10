from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
from dotenv import load_dotenv

load_dotenv()


class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        prompt = example['prompt']
        chosen = example['chosen']
        rejected = example['rejected']
        
        # Tokenize prompt + chosen response
        chosen_text = f"{prompt} [SEP] {chosen}"
        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize prompt + rejected response
        rejected_text = f"{prompt} [SEP] {rejected}"
        rejected_encoding = self.tokenizer(
            rejected_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'chosen_input_ids': chosen_encoding['input_ids'].squeeze(0),
            'chosen_attention_mask': chosen_encoding['attention_mask'].squeeze(0),
            'rejected_input_ids': rejected_encoding['input_ids'].squeeze(0),
            'rejected_attention_mask': rejected_encoding['attention_mask'].squeeze(0),
            'chosen_rating': example['chosen-rating'],
            'rejected_rating': example['rejected-rating'],
            'source': example['source']
        }


def prepare_data_splits(dataset_name="argilla/ultrafeedback-binarized-preferences-cleaned", 
                       train_ratio=0.7, 
                       val_ratio=0.15, 
                       test_ratio=0.15,
                       seed=42,
                       subsample_size=None):  # NEW: subsample parameter
    """
    Load dataset and create train/val/test splits
    
    Args:
        subsample_size: If provided, only use this many examples from the full dataset
    """
    # Load dataset
    ds = load_dataset(dataset_name)
    train_data = ds['train']
    
    # Subsample if requested
    if subsample_size is not None:
        print(f"Subsampling {subsample_size} examples from {len(train_data)} total")
        indices = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(seed))[:subsample_size]
        train_data = train_data.select(indices.tolist())
    
    print(f"Total dataset size: {len(train_data)}")
    
    # Calculate split sizes
    total_size = len(train_data)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    print(f"Split sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Create splits
    train_split, val_split, test_split = random_split(
        train_data, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_split, val_split, test_split


def create_dataloaders(train_split, val_split, test_split, 
                       tokenizer, 
                       batch_size=16, 
                       max_length=512,
                       num_workers=0):
    """
    Create PyTorch DataLoaders for train/val/test
    """
    train_dataset = PreferenceDataset(train_split, tokenizer, max_length)
    val_dataset = PreferenceDataset(val_split, tokenizer, max_length)
    test_dataset = PreferenceDataset(test_split, tokenizer, max_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader


def main():
    # Use smaller, faster model for Mac
    model_name = 'distilbert-base-uncased'  # 66M params, 2x faster than BERT
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    print(f"Successfully loaded: {model_name}")
    
    # Create data splits with subsampling
    train_split, val_split, test_split = prepare_data_splits(
        subsample_size=10000  # Use only 10K examples for fast local training
    )
    
    # Create dataloaders with smaller batch size for Mac
    train_loader, val_loader, test_loader = create_dataloaders(
        train_split, val_split, test_split,
        tokenizer,
        batch_size=8,  # Smaller batch size for Mac
        max_length=256  # Shorter sequences = faster training
    )
    
    # Test: Load one batch
    print("\nTesting data loading...")
    batch = next(iter(train_loader))
    
    print(f"Batch keys: {batch.keys()}")
    print(f"Chosen input_ids shape: {batch['chosen_input_ids'].shape}")
    print(f"Chosen attention_mask shape: {batch['chosen_attention_mask'].shape}")
    print(f"Rejected input_ids shape: {batch['rejected_input_ids'].shape}")
    print(f"Rejected attention_mask shape: {batch['rejected_attention_mask'].shape}")
    print(f"Chosen ratings: {batch['chosen_rating'][:3]}")
    print(f"Rejected ratings: {batch['rejected_rating'][:3]}")
    
    # Decode one example to verify
    print("\n--- Example from batch ---")
    chosen_text = tokenizer.decode(batch['chosen_input_ids'][0], skip_special_tokens=True)
    rejected_text = tokenizer.decode(batch['rejected_input_ids'][0], skip_special_tokens=True)
    print(f"Chosen text (first 200 chars): {chosen_text[:200]}...")
    print(f"Rejected text (first 200 chars): {rejected_text[:200]}...")
    print(f"Chosen rating: {batch['chosen_rating'][0].item()}")
    print(f"Rejected rating: {batch['rejected_rating'][0].item()}")
    
    print(f"\n✓ Data preparation successful with {model_name}!")
    print(f"✓ Using MPS-friendly settings: batch_size=8, max_length=256, 10K examples")


if __name__ == "__main__":
    main()