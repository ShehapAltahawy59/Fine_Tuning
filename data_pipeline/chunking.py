import re
def run(text, min_words=200, max_words=400):
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    for p in paragraphs:
        words = p.split()
        if min_words <= len(words) <= max_words:
            chunks.append(p.strip())
    return chunks
