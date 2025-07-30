import re

def run(text):
    # Merge broken lines
    text = re.sub(r'\n(?=[a-z])', ' ', text)  # join words split by line breaks
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)

    cleaned = re.sub(r'^[^A-Z]*', '', text)  
    # Remove multiple spaces/newlines
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

    return text.strip()
