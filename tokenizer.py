"""
Custom tokenizer for Sanskrit using grapheme clusters
Fixed version that properly handles Devanagari
"""

import regex
from typing import List, Dict, Tuple
import json
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SanskritGraphemeTokenizer:
    """
    Tokenizer that treats Sanskrit aksharas (grapheme clusters) as tokens
    """
    
    def __init__(self, vocab_size: int = 16000):
        self.vocab_size = vocab_size
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        
        # Initialize with special tokens
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to vocabulary"""
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for i, token in enumerate(special_tokens):
            self.stoi[token] = i
            self.itos[i] = token
    
    def _get_graphemes(self, text: str) -> List[str]:
        """
        Extract grapheme clusters from text
        \X matches a single grapheme cluster (akshara)
        """
        if not text:
            return []
        return regex.findall(r'\X', text)
    
    def train(self, texts: List[str], min_frequency: int = 2):
        """
        Build vocabulary from texts using grapheme clusters
        """
        logger.info(f"🔤 Training tokenizer on {len(texts)} texts...")
        
        # Count all grapheme clusters
        counter = Counter()
        total_graphemes = 0
        
        for text in texts:
            graphemes = self._get_graphemes(text)
            counter.update(graphemes)
            total_graphemes += len(graphemes)
        
        logger.info(f"Total graphemes found: {total_graphemes:,}")
        logger.info(f"Unique graphemes: {len(counter)}")
        
        # Filter by frequency
        filtered = {k: v for k, v in counter.items() if v >= min_frequency}
        logger.info(f"Graphemes with frequency >= {min_frequency}: {len(filtered)}")
        
        # Sort by frequency (descending)
        sorted_tokens = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        
        # Add to vocabulary (reserving space for special tokens)
        special_count = len(self.stoi)
        max_tokens = self.vocab_size - special_count
        
        # Add most frequent tokens
        added = 0
        for token, freq in sorted_tokens:
            if added >= max_tokens:
                break
            if token not in self.stoi:  # Avoid duplicates
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx] = token
                added += 1
        
        logger.info(f"✅ Built vocabulary with {len(self.stoi)} tokens")
        logger.info(f"  Special tokens: {special_count}")
        logger.info(f"  Sanskrit tokens: {len(self.stoi) - special_count}")
        
        # Log some examples
        sample_tokens = list(self.stoi.items())[:10]
        logger.info(f"  Sample tokens: {sample_tokens}")
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to token IDs
        """
        if not text:
            return []
        
        graphemes = self._get_graphemes(text)
        unk_id = self.stoi.get(self.unk_token, 0)
        
        token_ids = []
        for g in graphemes:
            if g in self.stoi:
                token_ids.append(self.stoi[g])
            else:
                # If token not found, try to split into smaller parts
                # This handles unknown characters gracefully
                for char in g:
                    if char in self.stoi:
                        token_ids.append(self.stoi[char])
                    else:
                        token_ids.append(unk_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Convert token IDs back to text
        """
        if not token_ids:
            return ""
        
        chars = []
        unk_token = self.unk_token
        
        for idx in token_ids:
            if idx in self.itos:
                token = self.itos[idx]
                # Skip special tokens
                if token in [self.pad_token, self.bos_token, self.eos_token]:
                    continue
                if token != unk_token:
                    chars.append(token)
                else:
                    chars.append('�')  # Replacement character for unknown
        
        return ''.join(chars)
    
    def save(self, path: str):
        """Save tokenizer to file"""
        data = {
            'stoi': self.stoi,
            'itos': {str(k): v for k, v in self.itos.items()},
            'vocab_size': self.vocab_size,
            'special_tokens': [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Tokenizer saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load tokenizer from file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.stoi = data['stoi']
        tokenizer.itos = {int(k): v for k, v in data['itos'].items()}
        
        logger.info(f"📂 Tokenizer loaded from {path}")
        logger.info(f"  Vocabulary size: {len(tokenizer.stoi)}")
        
        return tokenizer
    
    def get_vocab_size(self) -> int:
        """Get the current vocabulary size"""
        return len(self.stoi)