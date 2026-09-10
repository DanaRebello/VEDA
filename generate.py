"""
Text generation script for trained Sanskrit GPT
"""

import torch
import argparse
from pathlib import Path

from config import ModelConfig
from model import SanskritGPT
from tokenizer import SanskritGraphemeTokenizer

def generate_text(
    checkpoint_path: str,
    tokenizer_path: str,
    prompt: str = "",
    max_tokens: int = 500,
    temperature: float = 0.8,
    top_k: int = 50
):
    """
    Generate text from a trained model
    """
    # Load tokenizer
    tokenizer = SanskritGraphemeTokenizer.load(tokenizer_path)
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_config = checkpoint['model_config']
    model = SanskritGPT(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Encode prompt
    if prompt:
        tokens = tokenizer.encode(prompt)
        idx = torch.tensor([tokens], dtype=torch.long, device=device)
    else:
        bos_id = tokenizer.stoi[tokenizer.bos_token]
        idx = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(idx, max_tokens, temperature, top_k)
    
    # Decode
    text = tokenizer.decode(generated[0].tolist())
    return text

def main():
    parser = argparse.ArgumentParser(description='Generate text from Sanskrit GPT')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, required=True, help='Path to tokenizer file')
    parser.add_argument('--prompt', type=str, default='', help='Prompt text')
    parser.add_argument('--max-tokens', type=int, default=500, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k sampling parameter')
    
    args = parser.parse_args()
    
    text = generate_text(
        args.checkpoint,
        args.tokenizer,
        args.prompt,
        args.max_tokens,
        args.temperature,
        args.top_k
    )
    
    print("\n" + "="*60)
    print("Generated Text")
    print("="*60)
    print(text)
    print("="*60)

if __name__ == "__main__":
    main()