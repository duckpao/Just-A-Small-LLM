import torch
import torch.nn as nn
from tokenizer import CharTokenizer
from model import MiniGPT

# -----------------------------
# CONFIG
# -----------------------------
BATCH_SIZE = 32
CONTEXT_LEN = 128
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD DATA
# -----------------------------
with open("data/train.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# -----------------------------
# BATCH SAMPLER
# -----------------------------
def get_batch():
    ix = torch.randint(0, len(data) - CONTEXT_LEN - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i+CONTEXT_LEN] for i in ix])
    y = torch.stack([data[i+1:i+CONTEXT_LEN+1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

# -----------------------------
# MODEL
# -----------------------------
model = MiniGPT(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    max_len=CONTEXT_LEN,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# -----------------------------
# TRAIN LOOP
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for step in range(100):
        x, y = get_batch()
        logits = model(x)

        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / 100
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "model.pt")
print("Training finished. Model saved.")
