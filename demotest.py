"""
Training script for Sanskrit GPT - Corrected for Devanagari grapheme tokenizer.
Data file expected at: data/devang.txt (relative to this script).
"""

import torch
import logging
import sys
import os
import json
from pathlib import Path

# --- Path setup: make imports work regardless of CWD ---
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Local imports ---
from config import ModelConfig, TrainingConfig, DataConfig
from model import SanskritGPT
from tokenizer import SanskritGraphemeTokenizer
from preprocess import SanskritPreprocessor, DatasetPreparer


def main():
    logger.info("=" * 60)
    logger.info("🚀 Sanskrit GPT Training (Devanagari Grapheme Tokenizer)")
    logger.info("=" * 60)
    logger.info(f"📂 Script directory: {SCRIPT_DIR}")

    # =========================================================
    # 1. CONFIGURATION
    # =========================================================
    model_config = ModelConfig()
    # Small model for quick testing
    model_config.n_embd = 128
    model_config.n_head = 4
    model_config.n_layer = 4
    model_config.block_size = 256
    model_config.dropout = 0.1
    model_config.bias = True
    model_config.vocab_size = 4000  # overwritten after tokenizer trains

    train_config = TrainingConfig()
    train_config.batch_size = 4
    train_config.learning_rate = 3e-4
    train_config.max_iters = 1000
    train_config.eval_interval = 100
    train_config.eval_iters = 5
    train_config.save_interval = 500
    train_config.grad_clip = 1.0
    train_config.weight_decay = 0.1
    train_config.beta1 = 0.9
    train_config.beta2 = 0.95

    data_config = DataConfig()

    # =========================================================
    # 2. RESOLVE DATA FILE PATH
    # =========================================================
    data_path = SCRIPT_DIR / "data" / "devang.txt"

    if not data_path.exists():
        logger.error(f"❌ Data file not found: {data_path}")
        data_dir = SCRIPT_DIR / "data"
        if data_dir.exists():
            logger.error(f"📁 Contents of {data_dir}:")
            for item in sorted(data_dir.iterdir()):
                marker = "📁" if item.is_dir() else "📄"
                logger.error(f"   {marker} {item.name}")
        else:
            logger.error(f"📁 data/ folder does not exist at {data_dir}")
        return

    data_config.data_path = str(data_path)
    file_size = data_path.stat().st_size
    logger.info(f"✅ Data file: {data_path}  ({file_size:,} bytes)")

    # =========================================================
    # 3. PREPROCESS DEVANAGARI TEXT
    # =========================================================
    logger.info("📖 Preprocessing Devanagari text...")
    preprocessor = SanskritPreprocessor()

    try:
        texts = preprocessor.process_file(
            str(data_path),
            {
                'remove_metadata': False,   # keep verse numbers (१.१.१)
                'min_text_length': 5,       # short Vedic lines OK
                'max_text_length': 50000,
            },
        )
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info(f"✅ Extracted {len(texts)} sentence(s)")

    if not texts:
        logger.error("❌ No texts extracted! Check UTF-8 encoding of devang.txt")
        return

    # Show sample
    logger.info("📝 Sample sentences:")
    for i, t in enumerate(texts[:3]):
        preview = t[:100] + "..." if len(t) > 100 else t
        logger.info(f"   {i + 1}. {preview}")

    # =========================================================
    # 4. TRAIN GRAPHEME TOKENIZER
    # =========================================================
    logger.info("🔤 Training grapheme tokenizer...")
    tokenizer = SanskritGraphemeTokenizer(vocab_size=model_config.vocab_size)
    tokenizer.train(texts)

    tokenizer_path = SCRIPT_DIR / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    # Update vocab size to match actual tokenizer
    model_config.vocab_size = len(tokenizer.stoi)
    logger.info(f"✅ Final vocab size = {model_config.vocab_size}")

    # --- Sanity check: round-trip on a real sentence ---
    logger.info("🔍 Round-trip verification:")
    test_sent = texts[0][:80]
    ids = tokenizer.encode(test_sent)
    decoded = tokenizer.decode(ids)
    if decoded == test_sent:
        logger.info(f"   ✅ Round-trip OK ({len(ids)} tokens)")
    else:
        logger.warning(f"   ⚠️  Round-trip mismatch!")
        logger.warning(f"      original: {test_sent}")
        logger.warning(f"      decoded:  {decoded}")

    # =========================================================
    # 5. PREPARE TRAIN / VAL SPLIT
    # =========================================================
    logger.info("📊 Preparing dataset...")
    try:
        train_data, val_data = DatasetPreparer.prepare_data(
            texts, tokenizer, model_config.block_size,
            train_split=data_config.train_split,
        )
    except Exception as e:
        logger.error(f"❌ Dataset preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # =========================================================
    # 6. BUILD MODEL
    # =========================================================
    logger.info("🏗️  Building model...")
    model = SanskritGPT(model_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f"✅ Device: {device}")

    if device.type == 'cuda':
        logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Model parameters: {n_params:,}")

    # =========================================================
    # 7. OPTIMIZER
    # =========================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay,
    )

    # =========================================================
    # 8. TRAINING LOOP
    # =========================================================
    logger.info("=" * 60)
    logger.info(f"🚀 Training for {train_config.max_iters} iters...")
    logger.info("=" * 60)

    steps = 0
    best_val = float('inf')
    checkpoint_path = SCRIPT_DIR / "model_final.pt"

    try:
        while steps < train_config.max_iters:
            # --- Get batch ---
            X, Y = DatasetPreparer.get_batch(
                train_data,
                model_config.block_size,
                train_config.batch_size,
                device,
            )

            # --- Forward ---
            logits, loss = model(X, Y)

            # --- Backward ---
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), train_config.grad_clip
            )
            optimizer.step()

            # --- Eval + log ---
            if steps % train_config.eval_interval == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for _ in range(train_config.eval_iters):
                        try:
                            Xv, Yv = DatasetPreparer.get_batch(
                                val_data,
                                model_config.block_size,
                                min(4, train_config.batch_size),
                                device,
                            )
                            _, vl = model(Xv, Yv)
                            val_losses.append(vl.item())
                        except Exception:
                            continue

                if val_losses:
                    avg_val = sum(val_losses) / len(val_losses)
                    best_val = min(best_val, avg_val)
                    logger.info(
                        f"Step {steps:5d} | "
                        f"Train: {loss.item():.4f} | "
                        f"Val: {avg_val:.4f} | "
                        f"Best: {best_val:.4f}"
                    )
                else:
                    logger.info(
                        f"Step {steps:5d} | Train: {loss.item():.4f}"
                    )
                model.train()

            # --- Periodic checkpoint ---
            if steps > 0 and steps % train_config.save_interval == 0:
                torch.save(
                    {
                        'model_config': model_config,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step': steps,
                        'best_val': best_val,
                    },
                    str(checkpoint_path),
                )
                logger.info(f"💾 Checkpoint saved at step {steps}")

            steps += 1

            # --- Early stopping ---
            if steps > 100 and loss.item() < 0.1:
                logger.info(f"✅ Early stop at step {steps} (loss < 0.1)")
                break

    except KeyboardInterrupt:
        logger.info("⏹️  Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()

    # =========================================================
    # 9. SAVE FINAL MODEL
    # =========================================================
    logger.info("💾 Saving final model...")
    torch.save(
        {
            'model_config': model_config,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step': steps,
            'best_val': best_val,
        },
        str(checkpoint_path),
    )
    logger.info(f"✅ Model saved: {checkpoint_path}")

    # =========================================================
    # 10. GENERATE SAMPLE (Devanagari)
    # =========================================================
    if steps > 50:
        logger.info("📝 Generating Devanagari sample...")
        model.eval()

        prompts = ["ब्रह्म", "देवा", "अग्निर्"]
        for prompt in prompts:
            try:
                tokens = tokenizer.encode(prompt)
                idx = torch.tensor(
                    [tokens], dtype=torch.long, device=device
                )
                with torch.no_grad():
                    out = model.generate(
                        idx,
                        max_tokens=80,
                        temperature=0.8,
                        top_k=50,
                    )
                text = tokenizer.decode(out[0].tolist())
                print("\n" + "=" * 60)
                print(f"Prompt: {prompt}")
                print("=" * 60)
                print(text)
                print("=" * 60)
            except Exception as e:
                logger.error(f"Generation failed for '{prompt}': {e}")

    logger.info("🎉 Training complete!")
    logger.info(f"   Tokenizer: {tokenizer_path}")
    logger.info(f"   Model:     {checkpoint_path}")
    logger.info("")
    logger.info("To generate text, run:")
    logger.info(
        f"   python generate.py "
        f"--checkpoint {checkpoint_path.name} "
        f"--tokenizer {tokenizer_path.name} "
        f"--prompt ब्रह्म"
    )


if __name__ == "__main__":
    main()