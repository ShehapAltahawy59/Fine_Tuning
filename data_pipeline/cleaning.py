import logging
import re
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def normalize_text(text):
        # Convert to lowercase and standardize terms
        text = text.lower()
        text = re.sub(r'\bev\b', 'electric vehicle', text)
        text = re.sub(r'\bcharging station\b', 'charging point', text)
        return text


def run(text):
    # Merge broken lines
    text = re.sub(r'\n(?=[a-z])', ' ', text)  # join words split by line breaks
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)

    cleaned = re.sub(r'^[^A-Z]*', '', text)  
    # Remove multiple spaces/newlines
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    norm=normalize_text(cleaned)

    return norm

    
