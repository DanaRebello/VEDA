"""
VEDA - Sanskrit Language Model Training
Fixed: tokenization, vocab coverage, corpus cleaning, generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import unicodedata
import re
import logging
import math
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  1. CORPUS CLEANING
# ─────────────────────────────────────────────

def clean_sanskrit_text(text: str) -> str:
    """
    Keep only Devanagari, IAST-extended Latin, spaces, and newlines.
    Strip digits, punctuation noise, and stray ASCII.
    """
    # Normalize unicode (NFC keeps composed Devanagari intact)
    text = unicodedata.normalize("NFC", text)

    # Remove verse numbering patterns like "1.2", "||12||", "(3)"
    text = re.sub(r"\|\|?\s*\d+[\.\d]*\s*\|\|?", " ", text)
    text = re.sub(r"\(\s*\d+\s*\)", " ", text)
    text = re.sub(r"\b\d+[\.\d]*\b", " ", text)

    # Keep Devanagari block (U+0900–U+097F) + basic Latin letters + space/newline
    # Also keep Vedic extensions (U+1CD0–U+1CFF) if present
    allowed = re.compile(
        r"[^\u0900-\u097F\u1CD0-\u1CFF\u0020\u000A"   # Devanagari + Vedic + space/newline
        r"a-zA-Zāīūṛṝḷḹṃḥśṣṭḍṇñṅ"                    # IAST transliteration
        r"ĀĪŪṚṜḶḸṂḤŚṢṬḌṆÑṄ]",                         # IAST uppercase
        re.UNICODE
    )
    text = allowed.sub(" ", text)

    # Collapse multiple spaces; strip empty lines
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────
#  2. TOKENIZER  (character-level, Unicode-safe)
# ─────────────────────────────────────────────

class SanskritCharTokenizer:
    """
    Pure character-level tokenizer that handles:
    - Devanagari akshara (base + matra as a UNIT)
    - IAST Latin characters
    - Whitespace as a single token
    - No <UNK> for any character in the corpus
    """

    PAD = "<PAD>"
    BOS = "<BOS>"
    EOS = "<EOS>"
    SPECIAL = [PAD, BOS, EOS]

    def __init__(self):
        self.char2id: dict[str, int] = {}
        self.id2char: dict[int, str] = {}

    def build_vocab(self, text: str):
        # Build character set from actual corpus — zero UNK that way
        unique_chars = sorted(set(text), key=lambda c: ord(c))
        vocab = self.SPECIAL + unique_chars
        self.char2id = {c: i for i, c in enumerate(vocab)}
        self.id2char = {i: c for c, i in self.char2id.items()}
        logger.info(f"Vocab size: {len(vocab)} | "
                    f"Devanagari chars: {sum(1 for c in unique_chars if '\u0900' <= c <= '\u097F')}")

    def encode(self, text: str, add_bos=True, add_eos=True) -> list[int]:
        ids = []
        if add_bos:
            ids.append(self.char2id[self.BOS])
        for ch in text:
            ids.append(self.char2id.get(ch, self.char2id[self.PAD]))
        if add_eos:
            ids.append(self.char2id[self.EOS])
        return ids

    def decode(self, ids: list[int], skip_special=True) -> str:
        special_ids = {self.char2id[s] for s in self.SPECIAL if s in self.char2id}
        chars = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            chars.append(self.id2char.get(i, ""))
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.char2id)

    def save(self, path: str):
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.char2id, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "SanskritCharTokenizer":
        import json
        tok = cls()
        with open(path, encoding="utf-8") as f:
            tok.char2id = json.load(f)
        tok.id2char = {int(i): c for c, i in tok.char2id.items()}
        return tok


# ─────────────────────────────────────────────
#  3. DATASET
# ─────────────────────────────────────────────

class SanskritDataset(Dataset):
    def __init__(self, token_ids: list[int], seq_len: int):
        self.seq_len = seq_len
        # Chunk into (input, target) pairs with stride = seq_len
        self.samples = []
        for i in range(0, len(token_ids) - seq_len, seq_len):
            x = token_ids[i      : i + seq_len]
            y = token_ids[i + 1  : i + seq_len + 1]
            self.samples.append((torch.tensor(x, dtype=torch.long),
                                  torch.tensor(y, dtype=torch.long)))

    def __len__(self):  return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


# ─────────────────────────────────────────────
#  4. MODEL  (small Transformer)
# ─────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        def reshape(t):
            return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        q, k, v = map(reshape, (q, k, v))
        # Causal mask
        scale = math.sqrt(self.d_k)
        attn = (q @ k.transpose(-2, -1)) / scale
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = self.drop(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_mult=4, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class VEDAModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4,
                 seq_len=256, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embed   = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks  = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.embed.weight

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.embed(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, tokenizer, prompt: str, max_new=200,
                 temperature=0.8, top_k=40) -> str:
        self.eval()
        device = next(self.parameters()).device
        ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        eos_id = tokenizer.char2id[tokenizer.EOS]

        for _ in range(max_new):
            idx_cond = idx[:, -self.seq_len:]
            logits = self(idx_cond)[:, -1, :]         # (1, vocab)
            logits /= temperature
            # Top-k sampling
            if top_k > 0:
                top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_vals[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            if next_id.item() == eos_id:
                break
            idx = torch.cat([idx, next_id], dim=1)

        generated_ids = idx[0, len(ids):].tolist()
        return tokenizer.decode(generated_ids)


# ─────────────────────────────────────────────
#  5. TRAINING
# ─────────────────────────────────────────────

def train(
    corpus_path: str,
    save_dir: str = "./veda_checkpoints",
    seq_len: int = 256,
    batch_size: int = 32,
    max_steps: int = 5000,
    eval_interval: int = 100,
    lr: float = 3e-4,
    d_model: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # --- Load & clean corpus ---
    logger.info(f"Loading corpus from {corpus_path}")
    raw = Path(corpus_path).read_text(encoding="utf-8")
    text = clean_sanskrit_text(raw)
    logger.info(f"Corpus: {len(raw):,} → {len(text):,} chars after cleaning")

    # --- Tokenizer ---
    tokenizer = SanskritCharTokenizer()
    tokenizer.build_vocab(text)
    tokenizer.save(f"{save_dir}/tokenizer.json")

    # --- Encode full corpus ---
    all_ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    logger.info(f"Total tokens: {len(all_ids):,}")

    # --- Train / val split ---
    split = int(0.9 * len(all_ids))
    train_ids, val_ids = all_ids[:split], all_ids[split:]

    train_ds = SanskritDataset(train_ids, seq_len)
    val_ds   = SanskritDataset(val_ids,   seq_len)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=True)

    # --- Model ---
    model = VEDAModel(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, seq_len=seq_len,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {param_count:,} | Device: {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)

    # --- Training loop ---
    model.train()
    train_iter = iter(train_dl)
    best_val = float("inf")

    for step in range(1, max_steps + 1):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % eval_interval == 0:
            # Eval
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vx, vy in val_dl:
                    vx, vy = vx.to(device), vy.to(device)
                    vl = F.cross_entropy(
                        model(vx).view(-1, tokenizer.vocab_size), vy.view(-1)
                    )
                    val_losses.append(vl.item())
                    if len(val_losses) >= 20:
                        break
            val_loss = sum(val_losses) / len(val_losses)
            logger.info(f"Step {step:5d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), f"{save_dir}/best_model.pt")

            model.train()

    logger.info("🎉 Training complete!")

    # --- Sample generation ---
    logger.info("📝 Generating sample...")
    model.load_state_dict(torch.load(f"{save_dir}/best_model.pt", map_location=device))

    # Use a real Sanskrit prompt if corpus is Devanagari
    prompt = "अ"   # ← change to any starting syllable from your corpus
    sample = model.generate(tokenizer, prompt, max_new=200, temperature=0.8, top_k=40)

    print("\n" + "=" * 50)
    print("Prompt:", prompt)
    print("Generated Text:")
    print("=" * 50)
    print(sample)
    print("=" * 50)


# ─────────────────────────────────────────────
#  6. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VEDA Sanskrit LM")
    parser.add_argument("--corpus",   required=True, help="Path to .txt corpus file")
    parser.add_argument("--save_dir", default="./veda_checkpoints")
    parser.add_argument("--steps",    type=int, default=5000)
    parser.add_argument("--batch",    type=int, default=32)
    parser.add_argument("--seq_len",  type=int, default=256)
    parser.add_argument("--d_model",  type=int, default=256)
    parser.add_argument("--layers",   type=int, default=4)
    parser.add_argument("--heads",    type=int, default=4)
    parser.add_argument("--lr",       type=float, default=3e-4)
    args = parser.parse_args()

    train(
        corpus_path=args.corpus,
        save_dir=args.save_dir,
        max_steps=args.steps,
        batch_size=args.batch,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        lr=args.lr,
    )