import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
import os
from data_prep import prepare_data_splits, create_dataloaders


class RewardModel(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', pooling='cls'):
        super().__init__()
        # Load pretrained transformer
        self.transformer = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        
        # reward head
        hidden_size = self.transformer.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Pool the outputs
        if self.pooling == 'cls':
            # Use [CLS] token representation
            pooled = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == 'mean':
            # Mean pooling over all tokens
            pooled = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)
        
        # Get scalar reward score
        score = self.reward_head(pooled)
        return score.squeeze(-1)


def bradley_terry_loss(score_chosen, score_rejected):
    return -torch.log(torch.sigmoid(score_chosen - score_rejected)).mean()


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        chosen_input_ids = batch['chosen_input_ids'].to(device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(device)
        rejected_input_ids = batch['rejected_input_ids'].to(device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(device)
        
        # Get scores
        score_chosen = model(chosen_input_ids, chosen_attention_mask)
        score_rejected = model(rejected_input_ids, rejected_attention_mask)
        
        # Compute loss
        loss = bradley_terry_loss(score_chosen, score_rejected)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        correct += (score_chosen > score_rejected).sum().item()
        total += len(score_chosen)
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Acc: {correct/total:.4f}")
    
    return total_loss / len(train_loader), correct / total


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            chosen_input_ids = batch['chosen_input_ids'].to(device)
            chosen_attention_mask = batch['chosen_attention_mask'].to(device)
            rejected_input_ids = batch['rejected_input_ids'].to(device)
            rejected_attention_mask = batch['rejected_attention_mask'].to(device)
            
            # Get scores
            score_chosen = model(chosen_input_ids, chosen_attention_mask)
            score_rejected = model(rejected_input_ids, rejected_attention_mask)
            
            # Compute loss
            loss = bradley_terry_loss(score_chosen, score_rejected)
            
            # Track metrics
            total_loss += loss.item()
            correct += (score_chosen > score_rejected).sum().item()
            total += len(score_chosen)
    
    return total_loss / len(val_loader), correct / total


def main():
    # try to set GPU here from EC2 if possible. 
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Mac GPU)")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    model_name = 'distilbert-base-uncased'
    batch_size = 8
    max_length = 256
    num_epochs = 3
    learning_rate = 2e-5
    subsample_size = None  
    
    print(f"\n=== Training Configuration ===")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Max length: {max_length}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Subsample size: {subsample_size if subsample_size else 'Full dataset'}")
    print("=" * 50)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # Prepare data
    print("\n=== Preparing Data ===")
    train_split, val_split, test_split = prepare_data_splits(subsample_size=subsample_size)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_split, val_split, test_split,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length
    )
    
    # Initialize model
    print("\n=== Initializing Model ===")
    model = RewardModel(model_name=model_name, pooling='cls')
    model = model.to(device)
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\n=== Training ===")
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pt')
            print(f" Saved best model with val acc: {val_acc:.4f}")
    
    # Final test evaluation
    print("\n=== Final Test Evaluation ===")
    model.load_state_dict(torch.load('best_model.pt'))
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    
    print("\n=== Training Complete! ===")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Model saved to: best_model.pt")


if __name__ == "__main__":
    main()