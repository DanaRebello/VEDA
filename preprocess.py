"""
Data preprocessing for Sanskrit text - Complete Production Version
Handles multiple formats: GRETIL, DCS, prose, verse, mixed scripts
"""

import re
import unicodedata
from typing import List, Optional, Dict, Tuple
import numpy as np
from pathlib import Path
import logging
from collections import Counter
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SanskritPreprocessor:
    """
    Comprehensive preprocessor for Sanskrit text from various sources
    Handles: GRETIL, DCS, prose, verse, mixed Devanagari/IAST
    """
    
    def __init__(self):
        # Patterns for Sanskrit text
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]')
        self.sanskrit_pattern = re.compile(r'[\u0900-\u097F\s।॥a-zA-Z0-9\-]')
        self.verse_number_pattern = re.compile(r'^\d+[\.\)]?\s*$')
        self.metadata_patterns = [
            r'^.*?---.*?---.*$',  # YAML frontmatter
            r'^\s*<.*?>\s*$',     # XML/HTML tags
            r'^\s*\[.*?\]\s*$',   # Metadata in brackets
            r'^.*?:.*?$',         # Key-value pairs at line start
            r'^#.*$',             # Comments
            r'^[A-Z][A-Z\s]+$',   # All caps headers
            r'^[0-9]+\.\s*$',     # Numbered lists
            r'^\{.*\}$',          # JSON-like structures
            r'^@.*$',             # Annotations
        ]
        
    def normalize_text(self, text: str) -> str:
        """Normalize Unicode text to NFC form"""
        return unicodedata.normalize('NFC', text)
    
    def detect_encoding(self, filepath: str) -> str:
        """Detect file encoding"""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'utf-16', 'utf-16-le', 'utf-16-be']
        
        for encoding in encodings:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    f.read()
                return encoding
            except:
                continue
        
        return 'utf-8'  # Default
    
    def remove_metadata(self, text: str) -> str:
        """Remove common metadata patterns from text files"""
        lines = text.split('\n')
        cleaned_lines = []
        skip_next = False
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip empty lines at start
            if not cleaned_lines and not line_stripped:
                continue
            
            # Check if line matches any metadata pattern
            should_skip = False
            
            # Skip very short lines that are likely metadata
            if len(line_stripped) < 3 and line_stripped and not any(c in line_stripped for c in ['।', '॥']):
                should_skip = True
            
            # Check metadata patterns
            for pattern in self.metadata_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    should_skip = True
                    break
            
            # Skip lines that are just numbers
            if line_stripped.isdigit() or re.match(r'^\d+[\.\)]?\s*$', line_stripped):
                should_skip = True
            
            # Skip lines that are just special characters
            if len(line_stripped) > 0 and all(c in '।॥' for c in line_stripped):
                should_skip = True
            
            if not should_skip:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def clean_text(self, text: str, options: dict) -> str:
        """Clean Sanskrit text"""
        # Remove non-Sanskrit characters if option is set
        if options.get('remove_non_sanskrit', True):
            # Keep Devanagari, Latin, and important punctuation
            text = re.sub(r'[^\u0900-\u097F\s।॥a-zA-Z0-9\-\.\,\?\!;:()]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix common OCR/encoding issues
        text = text.replace('|', '।')  # Common OCR mistake
        text = text.replace('||', '॥')
        
        return text
    
    def split_by_danda(self, text: str) -> List[str]:
        """Split text using Sanskrit danda (। and ॥)"""
        # Replace double danda with single for splitting
        temp = text.replace('॥', '।')
        sentences = temp.split('।')
        return [s.strip() for s in sentences if s.strip()]
    
    def split_by_period(self, text: str) -> List[str]:
        """Split text using periods and other Western punctuation"""
        sentences = re.split(r'[.!?;:]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def split_by_newline(self, text: str) -> List[str]:
        """Split text by newlines (verse format)"""
        lines = text.split('\n')
        verses = []
        current_verse = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                if current_verse:
                    verses.append(' '.join(current_verse))
                    current_verse = []
                continue
            
            # Check if line is a verse number
            if re.match(r'^\d+[\.\)]?\s*$', line):
                if current_verse:
                    verses.append(' '.join(current_verse))
                    current_verse = []
                continue
            
            # Check if line starts with a verse number
            verse_match = re.match(r'^(\d+[\.\)]?)\s+(.*)', line)
            if verse_match:
                if current_verse:
                    verses.append(' '.join(current_verse))
                    current_verse = []
                current_verse.append(verse_match.group(2))
            else:
                current_verse.append(line)
        
        # Don't forget the last verse
        if current_verse:
            verses.append(' '.join(current_verse))
        
        return [v.strip() for v in verses if v.strip()]
    
    def create_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Create overlapping chunks of text"""
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i+chunk_size]
            if len(chunk) > 100:
                chunks.append(chunk.strip())
        return chunks
    
    def split_into_sentences(self, text: str, options: dict) -> List[str]:
        """
        Split text into sentences using multiple strategies
        Returns a list of clean sentences
        """
        all_sentences = []
        
        # Strategy 1: Split on Sanskrit punctuation (। and ॥)
        if '।' in text or '॥' in text:
            sent = self.split_by_danda(text)
            if sent:
                all_sentences.extend(sent)
                logger.debug(f"  Danda split: {len(sent)} sentences")
        
        # Strategy 2: Split on periods and other punctuation
        if not all_sentences or len(all_sentences) < 5:
            sent = self.split_by_period(text)
            if sent:
                all_sentences.extend(sent)
                logger.debug(f"  Period split: {len(sent)} sentences")
        
        # Strategy 3: Split on newlines (verse format)
        if not all_sentences or len(all_sentences) < 5:
            sent = self.split_by_newline(text)
            if sent:
                all_sentences.extend(sent)
                logger.debug(f"  Newline split: {len(sent)} sentences")
        
        # Strategy 4: Fallback - create chunks
        if not all_sentences or len(all_sentences) < 3:
            chunk_size = options.get('chunk_size', 500)
            overlap = options.get('overlap', 50)
            sent = self.create_chunks(text, chunk_size, overlap)
            if sent:
                all_sentences.extend(sent)
                logger.debug(f"  Chunk split: {len(sent)} chunks")
        
        # Clean and filter sentences
        cleaned_sentences = []
        min_len = options.get('min_text_length', 20)
        max_len = options.get('max_text_length', 10000)
        
        for sent in all_sentences:
            sent = sent.strip()
            if not sent:
                continue
            
            # Remove extra whitespace
            sent = re.sub(r'\s+', ' ', sent)
            
            # Filter by length
            if not (min_len <= len(sent) <= max_len):
                continue
            
            # Check if it has meaningful content
            devanagari_count = sum(1 for c in sent if '\u0900' <= c <= '\u097F')
            latin_words = len(re.findall(r'[a-zA-Z]{3,}', sent))
            total_words = len(sent.split())
            
            # Keep if it has enough content
            keep = False
            
            # Has significant Devanagari
            if devanagari_count > 5:
                keep = True
            # Has significant Latin words (IAST transliteration)
            elif latin_words > 3:
                keep = True
            # Has enough total words
            elif total_words > 5:
                keep = True
            # Has some Devanagari and some words
            elif devanagari_count > 0 and total_words > 3:
                keep = True
            
            if keep:
                cleaned_sentences.append(sent)
        
        return cleaned_sentences
    
    def analyze_text_structure(self, text: str) -> Dict:
        """Analyze text to determine best splitting strategy"""
        logger.info("📊 Analyzing text structure...")
        
        lines = text.split('\n')
        stats = {
            'total_chars': len(text),
            'total_lines': len(lines),
            'devanagari_chars': sum(1 for c in text if '\u0900' <= c <= '\u097F'),
            'latin_chars': sum(1 for c in text if c.isalpha() and ord(c) < 128),
            'danda_count': text.count('।'),
            'double_danda_count': text.count('॥'),
            'period_count': text.count('.'),
            'question_count': text.count('?'),
            'exclamation_count': text.count('!'),
            'newline_count': text.count('\n'),
            'avg_line_length': sum(len(line) for line in lines) / max(1, len(lines)),
            'empty_lines': sum(1 for line in lines if not line.strip()),
            'verse_lines': sum(1 for line in lines if re.match(r'^\d+[\.\)]?\s', line.strip())),
        }
        
        # Determine format
        if stats['danda_count'] > 50:
            stats['format'] = 'prose_with_danda'
        elif stats['newline_count'] > 50 and stats['verse_lines'] > 10:
            stats['format'] = 'verse'
        elif stats['period_count'] > 50:
            stats['format'] = 'prose_with_periods'
        elif stats['newline_count'] > 100:
            stats['format'] = 'verse_simple'
        else:
            stats['format'] = 'unknown'
        
        # Log statistics
        logger.info(f"  Format: {stats['format']}")
        logger.info(f"  Total characters: {stats['total_chars']:,}")
        logger.info(f"  Devanagari chars: {stats['devanagari_chars']:,}")
        logger.info(f"  Latin chars: {stats['latin_chars']:,}")
        logger.info(f"  Danda (।): {stats['danda_count']}")
        logger.info(f"  Periods (.): {stats['period_count']}")
        logger.info(f"  Newlines: {stats['newline_count']}")
        logger.info(f"  Verse lines: {stats['verse_lines']}")
        logger.info(f"  Average line length: {stats['avg_line_length']:.1f}")
        
        return stats
    
    def process_file(self, filepath: str, options: dict) -> List[str]:
        """Process a single text file"""
        logger.info(f"📖 Processing: {filepath}")
        
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Try different encodings
        content = None
        used_encoding = None
        
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'utf-16']:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                used_encoding = encoding
                logger.info(f"✅ Successfully read with {encoding}")
                break
            except Exception as e:
                continue
        
        if content is None:
            # Try binary read as last resort
            try:
                with open(filepath, 'rb') as f:
                    content = f.read().decode('utf-8', errors='ignore')
                used_encoding = 'utf-8 (binary)'
                logger.info(f"✅ Successfully read with binary fallback")
            except:
                raise ValueError(f"Could not read file: {filepath}")
        
        # Log initial stats
        logger.info(f"Original size: {len(content):,} characters")
        
        # Analyze text structure
        stats = self.analyze_text_structure(content)
        
        # Normalize
        content = self.normalize_text(content)
        
        # Remove metadata if option is set
        if options.get('remove_metadata', True):
            original_len = len(content)
            content = self.remove_metadata(content)
            removed = original_len - len(content)
            if removed > 0:
                logger.info(f"Removed {removed:,} metadata characters")
        
        # Clean text
        content = self.clean_text(content, options)
        logger.info(f"After cleaning: {len(content):,} characters")
        
        # Split into sentences
        sentences = self.split_into_sentences(content, options)
        logger.info(f"📝 Extracted {len(sentences)} sentences")
        
        # Log sample sentences
        if sentences:
            logger.info(f"📝 Sample sentences:")
            for i in range(min(5, len(sentences))):
                preview = sentences[i][:100] + "..." if len(sentences[i]) > 100 else sentences[i]
                logger.info(f"  {i+1}. {preview}")
        else:
            logger.warning("⚠️ No sentences extracted! Creating chunks as fallback...")
            # Try chunking as last resort
            sentences = self.create_chunks(content, chunk_size=500, overlap=50)
            logger.info(f"Created {len(sentences)} chunks")
        
        return sentences

class DatasetPreparer:
    """Prepare dataset for training"""
    
    @staticmethod
    def prepare_data(texts: List[str], tokenizer, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Convert texts to token IDs and create train/val split"""
        if not texts:
            raise ValueError("No texts provided for preparation!")
        
        logger.info(f"📊 Preparing dataset from {len(texts)} texts...")
        
        # Encode all texts
        encoded = []
        total_tokens = 0
        too_long = 0
        
        for i, text in enumerate(texts):
            try:
                tokens = tokenizer.encode(text)
                if len(tokens) > 0:
                    # Truncate very long texts to prevent issues
                    if len(tokens) > block_size * 4:
                        tokens = tokens[:block_size * 4]
                        too_long += 1
                    encoded.append(tokens)
                    total_tokens += len(tokens)
            except Exception as e:
                logger.warning(f"Error encoding text {i}: {e}")
                continue
        
        if not encoded:
            raise ValueError("No valid tokens after encoding!")
        
        # Log encoding stats
        logger.info(f"  Encoded {len(encoded)} texts")
        logger.info(f"  Total tokens: {total_tokens:,}")
        logger.info(f"  Average tokens per text: {total_tokens // len(encoded):,}")
        if too_long > 0:
            logger.info(f"  Truncated {too_long} texts")
        
        # Concatenate all texts with EOS separator
        all_ids = []
        eos_id = tokenizer.stoi.get(tokenizer.eos_token, 0)
        
        for tokens in encoded:
            all_ids.extend(tokens)
            all_ids.append(eos_id)
        
        # Convert to numpy array
        data = np.array(all_ids, dtype=np.int32)
        
        # Split into train/val
        n = len(data)
        train_size = int(n * 0.9)
        
        train_data = data[:train_size]
        val_data = data[train_size:]
        
        print(f"\n📊 Dataset statistics:")
        print(f"  Total texts: {len(texts):,}")
        print(f"  Total tokens: {n:,}")
        print(f"  Train tokens: {len(train_data):,}")
        print(f"  Val tokens: {len(val_data):,}")
        print(f"  Train batches: {len(train_data) // block_size:,}")
        print(f"  Val batches: {len(val_data) // block_size:,}")
        
        return train_data, val_data
    
    @staticmethod
    def get_batch(data, block_size: int, batch_size: int, device: str):
        """Get a batch of data for training"""
        import torch
        
        # Ensure we have enough data
        if len(data) < block_size + 1:
            raise ValueError(f"Data too short! Need {block_size + 1} tokens, have {len(data)}")
        
        # Random starting points
        ix = torch.randint(0, len(data) - block_size, (batch_size,))
        
        # Create batches
        x = torch.stack([torch.tensor(data[i:i+block_size], dtype=torch.long) for i in ix])
        y = torch.stack([torch.tensor(data[i+1:i+block_size+1], dtype=torch.long) for i in ix])
        
        return x.to(device), y.to(device)
    
    @staticmethod
    def get_batch_sequential(data, block_size: int, batch_size: int, device: str, start_idx: int):
        """Get a sequential batch (useful for evaluation)"""
        import torch
        
        end_idx = min(start_idx + batch_size, len(data) - block_size)
        actual_batch_size = end_idx - start_idx
        
        if actual_batch_size <= 0:
            return None, None
        
        x = torch.stack([torch.tensor(data[i:i+block_size], dtype=torch.long) for i in range(start_idx, end_idx)])
        y = torch.stack([torch.tensor(data[i+1:i+block_size+1], dtype=torch.long) for i in range(start_idx, end_idx)])
        
        return x.to(device), y.to(device)

def process_directory(input_dir: str, output_file: str, options: dict) -> List[str]:
    """Process all text files in a directory"""
    all_texts = []
    input_path = Path(input_dir)
    
    logger.info(f"📁 Processing directory: {input_dir}")
    
    # Process .txt files
    txt_files = list(input_path.glob('*.txt'))
    logger.info(f"Found {len(txt_files)} text files")
    
    preprocessor = SanskritPreprocessor()
    
    for filepath in txt_files:
        try:
            texts = preprocessor.process_file(str(filepath), options)
            all_texts.extend(texts)
            logger.info(f"  ✅ {filepath.name}: {len(texts)} texts")
        except Exception as e:
            logger.error(f"  ❌ {filepath.name}: {e}")
    
    # Save corpus
    if all_texts and output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_texts))
        logger.info(f"💾 Saved corpus to {output_file}")
    
    logger.info(f"Total: {len(all_texts)} texts")
    return all_texts