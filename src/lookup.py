import re
import os

def lookup_verse(prompt, dataset_path='datasets/kjv.txt'):
    """
    Checks if the prompt is a specific Bible reference (e.g. "Genesis 1:1").
    If so, returns the exact text from the dataset.
    Returns None if no match is found.
    """
    # Pattern to match "Book [Optional Number] Chapter:Verse"
    # Examples: "Genesis 1:1", "1 Chronicles 7:23", "John 3:16"
    pattern = r'^([123]?\s?[A-Za-z]+)\s+(\d+):(\d+)$'
    match = re.search(pattern, prompt.strip(), re.IGNORECASE)
    
    if not match:
        return None
    
    book = match.group(1).strip()
    chapter = match.group(2)
    verse = match.group(3)
    
    # Standardize the search string to match the KJV file format: "Book Chapter:Verse"
    # The KJV file uses a tab between the reference and the text.
    search_ref = f"{book} {chapter}:{verse}".lower()
    
    if not os.path.exists(dataset_path):
        return None
        
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Check if the line starts with our reference
                if line.lower().startswith(search_ref):
                    # Return the full line (Reference + Tab + Text)
                    return line.strip()
    except Exception as e:
        print(f"Lookup error: {e}")
        
    return None
