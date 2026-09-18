"""
Sanskrit Grapheme Tokenizer for Devanagari script
Properly handles:
- Devanagari consonants + vowel signs (matras) as single tokens
- Virama/halant conjuncts
- Vedic accents (udatta, anudatta)
- Danda (।) and Double Danda (॥)
- Nukta, avagraha, and other combining marks
"""

import json
import re
from collections import Counter
from typing import List, Dict


class SanskritGraphemeTokenizer:
    """
    Tokenizer that treats Devanagari grapheme clusters as atomic tokens.
    This is crucial for Sanskrit because a 'character' like 'क्ष' (kṣa)
    is composed of multiple Unicode code points but is one visual unit.
    """

    # Unicode ranges
    DEVANAGARI_START = '\u0900'
    DEVANAGARI_END = '\u097F'
    DEVANAGARI_EXTENDED_START = '\uA8E0'  # Devanagari Extended (Vedic accents)
    DEVANAGARI_EXTENDED_END = '\uA8FF'
    VEDIC_EXTENSIONS_START = '\u1CD0'  # Vedic Extensions
    VEDIC_EXTENSIONS_END = '\u1CFF'

    # Combining marks (matras, virama, anusvara, visarga, etc.)
    # These MUST attach to the preceding character
    COMBINING_MARKS = set(
        '\u093A\u093B\u093C\u093E\u093F\u0940\u0941\u0942\u0943\u0944'
        '\u0945\u0946\u0947\u0948\u0949\u094A\u094B\u094C\u094D\u094E\u094F'
        '\u0951\u0952\u0953\u0954\u0955\u0956\u0957'
        '\u0962\u0963'
        '\uA8E0\uA8E1\uA8E2\uA8E3\uA8E4\uA8E5\uA8E6\uA8E7\uA8E8\uA8E9'
        '\uA8EA\uA8EB\uA8EC\uA8ED\uA8EE\uA8EF\uA8F0\uA8F1'
        '\u1CD0\u1CD1\u1CD2\u1CD3\u1CD4\u1CD5\u1CD6\u1CD7\u1CD8\u1CD9'
        '\u1CDA\u1CDB\u1CDC\u1CDD\u1CDE\u1CDF\u1CE0\u1CE1\u1CE2'
        '\u1CE3\u1CE4\u1CE5\u1CE6\u1CE7\u1CE8\u1CE9\u1CEA\u1CEB'
        '\u1CEC\u1CED\u1CEE\u1CEF\u1CF0\u1CF1\u1CF2\u1CF3\u1CF4'
        '\u1CF5\u1CF6\u1CF7\u1CF8\u1CF9'
    )

    # Virama (halant) - triggers conjunct formation
    VIRAMA = '\u094D'
    ZWJ = '\u200D'  # Zero Width Joiner
    ZWNJ = '\u200C'  # Zero Width Non-Joiner

    # Special tokens
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'
    BOS_TOKEN = '<bos>'
    EOS_TOKEN = '<eos>'

    def __init__(self, vocab_size: int = 16000):
        self.vocab_size = vocab_size
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        self._trained = False

    def _is_devanagari(self, ch: str) -> bool:
        """Check if character is Devanagari or Vedic extension."""
        cp = ord(ch)
        return (
            (0x0900 <= cp <= 0x097F) or
            (0xA8E0 <= cp <= 0xA8FF) or
            (0x1CD0 <= cp <= 0x1CFF)
        )

    def _is_combining(self, ch: str) -> bool:
        """Check if character is a combining mark."""
        return ch in self.COMBINING_MARKS or ch in (
            self.VIRAMA, self.ZWJ, self.ZWNJ
        )

    def graphemize(self, text: str) -> List[str]:
        """
        Split Devanagari text into grapheme clusters.
        
        Rules:
        1. A base character (consonant/vowel) + any following combining marks = 1 token
        2. Consonant + Virama + Consonant = conjunct (1 token)
        3. Vedic accents attach to the preceding syllable
        4. Non-Devanagari characters are individual tokens
        
        Examples:
        - 'क' → ['क']
        - 'का' → ['का']  (ka + aa-matra)
        - 'क्ष' → ['क्ष']  (ka + virama + ṣa)
        - 'कृष्ण' → ['कृ', 'ष्ण']  (kṛ + ṣṇa)
        - 'सुब्रह्मण्यो३म्' → ['सु', 'ब्र', 'ह्म', 'ण्यो', '३', 'म्']
        """
        graphemes = []
        i = 0
        n = len(text)

        while i < n:
            ch = text[i]

            if not self._is_devanagari(ch):
                # Non-Devanagari: handle whitespace and punctuation separately
                graphemes.append(ch)
                i += 1
                continue

            # Start a new grapheme cluster with this Devanagari char
            cluster = [ch]
            i += 1

            # Consume following combining marks
            while i < n and self._is_combining(text[i]):
                # Special case: Virama followed by another consonant = conjunct
                if text[i] == self.VIRAMA and i + 1 < n and self._is_devanagari(text[i + 1]):
                    cluster.append(text[i])  # add virama
                    i += 1
                    # Check for ZWJ/ZWNJ after virama
                    while i < n and text[i] in (self.ZWJ, self.ZWNJ):
                        cluster.append(text[i])
                        i += 1
                    # Add the next consonant
                    if i < n and self._is_devanagari(text[i]):
                        cluster.append(text[i])
                        i += 1
                    # Continue consuming combining marks for this new consonant
                    continue
                else:
                    cluster.append(text[i])
                    i += 1

            # Consume Vedic accents and other modifiers
            while i < n and (
                '\u1CD0' <= text[i] <= '\u1CFF' or
                '\uA8E0' <= text[i] <= '\uA8FF' or
                text[i] in ('\u0951', '\u0952', '\u0953', '\u0954')
            ):
                cluster.append(text[i])
                i += 1

            graphemes.append(''.join(cluster))

        return graphemes

    def train(self, texts: List[str]):
        """Build vocabulary from grapheme clusters."""
        counter = Counter()

        for text in texts:
            graphemes = self.graphemize(text)
            counter.update(graphemes)

        # Start with special tokens
        self.stoi = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
            self.BOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
        }

        # Add most frequent graphemes
        for grapheme, _ in counter.most_common(self.vocab_size - len(self.stoi)):
            if grapheme not in self.stoi:
                self.stoi[grapheme] = len(self.stoi)

        # Build inverse mapping
        self.itos = {v: k for k, v in self.stoi.items()}
        self._trained = True

        print(f"✅ Tokenizer trained: {len(self.stoi)} tokens")
        print(f"   Top 20 graphemes: {[g for g, _ in counter.most_common(20)]}")

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode text to token IDs."""
        if not self._trained:
            raise RuntimeError("Tokenizer not trained. Call train() first.")

        graphemes = self.graphemize(text)
        ids = []
        if add_special_tokens:
            ids.append(self.stoi[self.BOS_TOKEN])
        for g in graphemes:
            ids.append(self.stoi.get(g, self.stoi[self.UNK_TOKEN]))
        if add_special_tokens:
            ids.append(self.stoi[self.EOS_TOKEN])
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        special = {self.stoi[t] for t in (
            self.PAD_TOKEN, self.UNK_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN
        )}
        chars = []
        for i in ids:
            if skip_special_tokens and i in special:
                continue
            chars.append(self.itos.get(i, ''))
        return ''.join(chars)

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab_size': self.vocab_size,
                'stoi': self.stoi,
                'itos': {str(k): v for k, v in self.itos.items()},
            }, f, ensure_ascii=False, indent=2)
        print(f"✅ Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tok = cls(vocab_size=data['vocab_size'])
        tok.stoi = data['stoi']
        tok.itos = {int(k): v for k, v in data['itos'].items()}
        tok._trained = True
        return tok