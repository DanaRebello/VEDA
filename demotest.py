"""
Training script for Sanskrit GPT - Debug version
"""

import torch
import logging
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import from your modules
from config import ModelConfig, TrainingConfig, DataConfig
from model import SanskritGPT
from tokenizer import SanskritGraphemeTokenizer
from preprocess import SanskritPreprocessor, DatasetPreparer

def main():
    logger.info("🚀 Starting training (debug version)...")
    
    # Load configuration - using smaller values for testing
    model_config = ModelConfig()
    model_config.n_embd = 128
    model_config.n_head = 4
    model_config.n_layer = 4
    model_config.block_size = 256
    model_config.vocab_size = 2000
    
    train_config = TrainingConfig()
    train_config.batch_size = 4
    train_config.max_iters = 1000
    train_config.eval_interval = 100
    train_config.save_interval = 500
    
    data_config = DataConfig()
    data_config.data_path = "data/SB.txt"
    
    # Check if file exists
    if not os.path.exists(data_config.data_path):
        logger.error(f"❌ File not found: {data_config.data_path}")
        logger.info(f"Current directory contents: {os.listdir('data') if os.path.exists('data') else 'data folder not found'}")
        return
    
    # Load and preprocess data
    logger.info("📖 Loading and preprocessing data...")
    preprocessor = SanskritPreprocessor()
    options = {
        'remove_metadata': True,
        'min_text_length': 10,
        'max_text_length': 10000,
        'remove_non_sanskrit': True
    }
    
    try:
        texts = preprocessor.process_file(data_config.data_path, options)
        logger.info(f"✅ Extracted {len(texts)} sentences")
        
        if not texts:
            logger.error("❌ No texts extracted! Trying alternative processing...")
            
            # Try reading raw text without splitting on danda
            with open(data_config.data_path, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            # Just take chunks of text
            chunk_size = 500
            texts = [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size) if len(raw_text[i:i+chunk_size]) > 100]
            logger.info(f"✅ Created {len(texts)} text chunks as fallback")
            
            if not texts:
                logger.error("❌ Still no texts! Please check your data file format.")
                return
                
    except Exception as e:
        logger.error(f"❌ Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create tokenizer
    logger.info("🔤 Training tokenizer...")
    tokenizer = SanskritGraphemeTokenizer(vocab_size=model_config.vocab_size)
    tokenizer.train(texts)
    tokenizer.save('tokenizer.json')
    logger.info(f"✅ Tokenizer trained with {len(tokenizer.stoi)} tokens")
    
    # Update vocab size
    model_config.vocab_size = len(tokenizer.stoi)
    logger.info(f"✅ Updated vocab size to {model_config.vocab_size}")
    
    # Prepare dataset
    try:
        train_data, val_data = DatasetPreparer.prepare_data(
            texts, tokenizer, model_config.block_size
        )
    except Exception as e:
        logger.error(f"❌ Error preparing dataset: {e}")
        return
    
    # Create model
    logger.info("🏗️ Building model...")
    model = SanskritGPT(model_config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f"✅ Using device: {device}")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay
    )
    
    # Training loop
    logger.info("🚀 Starting training...")
    steps = 0
    
    try:
        while steps < train_config.max_iters:
            # Get batch
            X, Y = DatasetPreparer.get_batch(
                train_data, 
                model_config.block_size, 
                train_config.batch_size,
                device
            )
            
            # Forward pass
            logits, loss = model(X, Y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
            
            # Log
            if steps % train_config.eval_interval == 0:
                # Calculate validation loss
                model.eval()
                with torch.no_grad():
                    val_losses = []
                    for _ in range(5):
                        try:
                            X_val, Y_val = DatasetPreparer.get_batch(
                                val_data, 
                                model_config.block_size, 
                                min(4, train_config.batch_size),
                                device
                            )
                            _, loss_val = model(X_val, Y_val)
                            val_losses.append(loss_val.item())
                        except:
                            continue
                    
                    if val_losses:
                        avg_val_loss = sum(val_losses) / len(val_losses)
                        logger.info(f"Step {steps:5d} | Train Loss: {loss.item():.4f} | Val Loss: {avg_val_loss:.4f}")
                    else:
                        logger.info(f"Step {steps:5d} | Train Loss: {loss.item():.4f}")
                
                model.train()
            
            steps += 1
            
            # Early stopping if loss is good
            if steps > 100 and loss.item() < 0.1:
                logger.info(f"✅ Early stopping at step {steps} - loss is very low!")
                break
    
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    
    # Save final model
    logger.info("💾 Saving model...")
    torch.save({
        'model_config': model_config,
        'model_state_dict': model.state_dict(),
    }, 'model_final.pt')
    logger.info("✅ Model saved!")
    
    # Generate sample if we have some training done
    if steps > 50:
        logger.info("📝 Generating sample...")
        model.eval()
        prompt = "रामः "
        try:
            tokens = tokenizer.encode(prompt)
            idx = torch.tensor([tokens], dtype=torch.long, device=device)
            
            with torch.no_grad():
                generated = model.generate(idx, max_tokens=100, temperature=0.8, top_k=50)
            
            text = tokenizer.decode(generated[0].tolist())
            print("\n" + "="*50)
            print("Generated Text:")
            print("="*50)
            print(text)
            print("="*50)
        except Exception as e:
            logger.error(f"Error generating text: {e}")
    
    logger.info("🎉 Training complete!")

if __name__ == "__main__":
    main()