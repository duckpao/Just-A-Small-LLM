import torch
import torch.nn.functional as F
from tokenizer import CharTokenizer
from model import MiniGPT

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "model.pt"
DATA_PATH = "data/train.txt"
CONTEXT_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# generation params
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.8   # lower = safer, higher = more random
TOP_K = 20          # limit randomness

# -----------------------------
# LOAD DATA & TOKENIZER
# -----------------------------
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = MiniGPT(
    vocab_size=tokenizer.vocab_size,
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    max_len=CONTEXT_LEN,
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -----------------------------
# GENERATION FUNCTION
# -----------------------------
@torch.no_grad()
def generate(prompt: str):
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=DEVICE).unsqueeze(0)

    for _ in range(MAX_NEW_TOKENS):
        # crop context if too long
        tokens_cond = tokens[:, -CONTEXT_LEN:]

        logits = model(tokens_cond)
        logits = logits[:, -1, :] / TEMPERATURE

        # top-k filtering
        if TOP_K is not None:
            v, ix = torch.topk(logits, TOP_K)
            logits_filtered = torch.full_like(logits, float("-inf"))
            logits_filtered.scatter_(1, ix, v)
            logits = logits_filtered

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        tokens = torch.cat([tokens, next_token], dim=1)

    return tokenizer.decode(tokens[0].tolist())


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    prompt = "[2024-01-01 10:00:00] ERROR service=api message="
    print("PROMPT:")
    print(prompt)
    print("\nGENERATED:\n")
    print(generate(prompt))
