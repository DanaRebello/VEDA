"""
Debug script to understand your text file structure
"""

import re

def analyze_text_file(filepath):
    print(f"🔍 Analyzing: {filepath}")
    print("="*50)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"Total characters: {len(content):,}")
    print(f"Total lines: {len(content.split('\n')):,}")
    
    # Check for different punctuation
    print(f"\n📊 Punctuation analysis:")
    print(f"  Danda (।): {content.count('।')}")
    print(f"  Double danda (॥): {content.count('॥')}")
    print(f"  Period (.): {content.count('.')}")
    print(f"  Question (?): {content.count('?')}")
    print(f"  Exclamation (!): {content.count('!')}")
    print(f"  Newlines: {content.count('\\n')}")
    
    # Show first 10 lines
    print(f"\n📝 First 10 lines:")
    print("="*50)
    lines = content.split('\n')[:10]
    for i, line in enumerate(lines):
        if line.strip():
            preview = line[:100] + "..." if len(line) > 100 else line
            print(f"{i+1:2d}. {preview}")
    
    # Show last 10 lines
    print(f"\n📝 Last 10 lines:")
    print("="*50)
    lines = content.split('\n')[-10:]
    for i, line in enumerate(lines):
        if line.strip():
            preview = line[:100] + "..." if len(line) > 100 else line
            print(f"{i+1:2d}. {preview}")
    
    # Try to identify format
    print(f"\n📌 Format identification:")
    if '।' in content and '॥' in content:
        print("  ✅ Contains Sanskrit danda punctuation")
    elif '।' in content:
        print("  ✅ Contains single danda")
    elif content.count('\n') > 20:
        print("  ✅ Likely verse format (many newlines)")
    elif content.count('.') > 10:
        print("  ✅ Likely prose with periods")
    else:
        print("  ⚠️ Unclear format - might need special processing")
    
    # Check for Devanagari
    devanagari_count = sum(1 for c in content if '\u0900' <= c <= '\u097F')
    latin_count = sum(1 for c in content if c.isalpha() and ord(c) < 128)
    
    print(f"\n📊 Character analysis:")
    print(f"  Devanagari characters: {devanagari_count:,}")
    print(f"  Latin characters: {latin_count:,}")
    print(f"  Devanagari percentage: {devanagari_count/len(content)*100:.1f}%")
    
    # Suggest splitting strategy
    print(f"\n💡 Suggested splitting strategy:")
    if '।' in content or '॥' in content:
        print("  Use danda splitting")
    elif content.count('\n') > len(content) / 100:
        print("  Use newline splitting (verse format)")
    elif '.' in content:
        print("  Use period splitting (prose)")
    else:
        print("  Use chunk-based splitting")

if __name__ == "__main__":
    analyze_text_file("data/SB.txt")