import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# ----------------- Dataset -----------------
class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        prompt = item['prompt']
        chosen_response = item['chosen'][1]['content']
        rejected_response = item['rejected'][1]['content']

        chosen_text = prompt + " " + chosen_response
        rejected_text = prompt + " " + rejected_response

        chosen_encoding = self.tokenizer(
            chosen_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
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
            'rejected_attention_mask': rejected_encoding['attention_mask'].squeeze(0)
        }


# ----------------- Reward Model -----------------
class RewardModel(nn.Module):
    """
    pooling options:
      - 'cls'   : CLS token
      - 'mean'  : mean pooling
      - 'attn'  : learned attention pooling
      - 'multi' : concat([CLS, mean]) -> bigger head
    """
    def __init__(self, model_name='distilbert-base-uncased',
                 pooling='cls', dropout=0.1):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

        hidden_size = self.transformer.config.hidden_size

        if pooling == 'attn':
            self.attn_vector = nn.Linear(hidden_size, 1)

        input_dim = hidden_size * (2 if pooling == 'multi' else 1)

        self.reward_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def _cls(self, last_hidden):
        return last_hidden[:, 0, :]

    def _mean(self, last_hidden, attention_mask):
        mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
        summed = (last_hidden * mask).sum(1)               # [B, H]
        counts = mask.sum(1).clamp(min=1e-6)               # [B, 1]
        return summed / counts

    def _attn(self, last_hidden, attention_mask):
        scores = self.attn_vector(last_hidden).squeeze(-1)   # [B, L]
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)              # [B, L]
        pooled = torch.bmm(weights.unsqueeze(1), last_hidden).squeeze(1)
        return pooled

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids,
                                   attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, L, H]

        if self.pooling == 'cls':
            pooled = self._cls(last_hidden)
        elif self.pooling == 'mean':
            pooled = self._mean(last_hidden, attention_mask)
        elif self.pooling == 'attn':
            pooled = self._attn(last_hidden, attention_mask)
        elif self.pooling == 'multi':
            cls_vec = self._cls(last_hidden)
            mean_vec = self._mean(last_hidden, attention_mask)
            pooled = torch.cat([cls_vec, mean_vec], dim=-1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        score = self.reward_head(pooled)
        return score.squeeze(-1)
def bradley_terry_loss(score_chosen, score_rejected):
    return -torch.log(torch.sigmoid(score_chosen - score_rejected)).mean()


def train_epoch(model, train_loader, optimizer, device, max_batches=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, batch in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        chosen_input_ids = batch['chosen_input_ids'].to(device)
        chosen_attention_mask = batch['chosen_attention_mask'].to(device)
        rejected_input_ids = batch['rejected_input_ids'].to(device)
        rejected_attention_mask = batch['rejected_attention_mask'].to(device)

        score_chosen = model(chosen_input_ids, chosen_attention_mask)
        score_rejected = model(rejected_input_ids, rejected_attention_mask)

        loss = bradley_terry_loss(score_chosen, score_rejected)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (score_chosen > score_rejected).sum().item()
        total += len(score_chosen)

    avg_loss = total_loss / max(1, (batch_idx + 1))
    acc = correct / max(1, total)
    return avg_loss, acc


def evaluate(model, val_loader, device, max_batches=None):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            chosen_input_ids = batch['chosen_input_ids'].to(device)
            chosen_attention_mask = batch['chosen_attention_mask'].to(device)
            rejected_input_ids = batch['rejected_input_ids'].to(device)
            rejected_attention_mask = batch['rejected_attention_mask'].to(device)

            score_chosen = model(chosen_input_ids, chosen_attention_mask)
            score_rejected = model(rejected_input_ids, rejected_attention_mask)

            loss = bradley_terry_loss(score_chosen, score_rejected)

            total_loss += loss.item()
            correct += (score_chosen > score_rejected).sum().item()
            total += len(score_chosen)

    avg_loss = total_loss / max(1, (batch_idx + 1))
    acc = correct / max(1, total)
    return avg_loss, acc
from itertools import product

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# ----------------- Base config -----------------
model_name = 'distilbert-base-uncased'
batch_size = 32
base_epochs = 3
subsample_size = 5000

# Hyperparameter grid
POOLINGS = ['cls', 'mean', 'attn', 'multi']
MAX_LENGTHS = [256, 384, 512]
LRS = [1e-5, 2e-5, 3e-5]
DROPOUTS = [0.1, 0.2]

# Compute safeguards
MAX_TOTAL_UPDATES = 40_000      # total training steps across all configs
MAX_BATCHES_PER_EPOCH = 300     # cap per epoch (for huge datasets)
EARLY_STOP_AFTER_EPOCH1_IF_ACC_BELOW = 0.52  # tweak as you like

total_updates_so_far = 0
results = []

# ----------------- Load dataset once -----------------
print("\n=== Loading UltraFeedback dataset ===")
ds = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")
dataset = ds['train']
if subsample_size:
    dataset = dataset.select(range(min(subsample_size, len(dataset))))

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
train_data = dataset.select(range(train_size))
val_data = dataset.select(range(train_size, train_size + val_size))
test_data = dataset.select(range(train_size + val_size, len(dataset)))

print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Precompute approximate steps per epoch (independent of max_length)
steps_per_epoch_full = math.ceil(len(train_data) / batch_size)
steps_per_epoch = min(steps_per_epoch_full, MAX_BATCHES_PER_EPOCH)

print(f"Full steps/epoch: {steps_per_epoch_full}, capped at: {steps_per_epoch}")
config_id = 0

for pooling, max_len, lr, dropout in product(POOLINGS, MAX_LENGTHS, LRS, DROPOUTS):
    config_id += 1
    est_updates = steps_per_epoch * base_epochs

    if total_updates_so_far + est_updates > MAX_TOTAL_UPDATES:
        print(f"\n[SKIP] Config {config_id}: {pooling}, L={max_len}, lr={lr}, drop={dropout} "
              f"(would exceed MAX_TOTAL_UPDATES)")
        continue

    print("\n" + "="*70)
    print(f"Config {config_id}: pooling={pooling}, max_len={max_len}, lr={lr}, dropout={dropout}")
    print("="*70)

    # Build datasets / loaders for this max_len
    train_dataset = PreferenceDataset(train_data, tokenizer, max_len)
    val_dataset = PreferenceDataset(val_data, tokenizer, max_len)
    test_dataset = PreferenceDataset(test_data, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Model + optimizer
    model = RewardModel(model_name=model_name, pooling=pooling, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(base_epochs):
        print(f"\n  Epoch {epoch+1}/{base_epochs}")
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device,
            max_batches=MAX_BATCHES_PER_EPOCH
        )
        val_loss, val_acc = evaluate(
            model, val_loader, device,
            max_batches=MAX_BATCHES_PER_EPOCH
        )
        print(f"    Train loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"    Val   loss={val_loss:.4f}, acc={val_acc:.4f}")

        # Early stop after first epoch for hopeless configs
        if epoch == 0 and val_acc < EARLY_STOP_AFTER_EPOCH1_IF_ACC_BELOW:
            print("    → Early stopping this config (val_acc below threshold).")
            break

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    # Estimate how many updates we actually used
    used_epochs = epoch + 1
    used_updates = steps_per_epoch * used_epochs
    total_updates_so_far += used_updates

    # Evaluate best model on test set (if we trained at least one epoch)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)
        test_loss, test_acc = evaluate(
            model, test_loader, device,
            max_batches=MAX_BATCHES_PER_EPOCH
        )
    else:
        test_loss, test_acc = None, None

    results.append({
        "config_id": config_id,
        "pooling": pooling,
        "max_len": max_len,
        "lr": lr,
        "dropout": dropout,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc) if test_acc is not None else None,
        "used_updates": used_updates,
    })

    print(f"\n  >>> Config {config_id} summary:")
    print(f"      best_val_acc = {best_val_acc:.4f}")
    print(f"      test_acc      = {test_acc:.4f}" if test_acc is not None else "      test_acc      = N/A")
    print(f"      updates used  = {used_updates}")
    print(f"      total updates so far = {total_updates_so_far} / {MAX_TOTAL_UPDATES}")

    if total_updates_so_far >= MAX_TOTAL_UPDATES:
        print("\nReached MAX_TOTAL_UPDATES – stopping grid search.")
        break

print("\n=== GRID SEARCH COMPLETE ===")
results_sorted = sorted(results, key=lambda x: x["best_val_acc"], reverse=True)
for r in results_sorted[:10]:
    print(f"Config {r['config_id']}: pool={r['pooling']}, "
          f"L={r['max_len']}, lr={r['lr']}, drop={r['dropout']} "
          f"→ val={r['best_val_acc']:.4f}, test={r['test_acc']}")

import pandas as pd

# Convert search results to a DataFrame
df_results = pd.DataFrame(results_sorted)

# Show pretty table in notebook
print("\n=== GRID SEARCH RESULTS TABLE (Top Configs) ===")
print(df_results.head(20).to_markdown(index=False))

# Save outputs
df_results.to_csv("grid_search_results.csv", index=False)
df_results.to_markdown("grid_search_results.md", index=False)

# LaTeX export (for research papers)
with open("grid_search_results.tex", "w") as f:
    f.write(df_results.to_latex(index=False))