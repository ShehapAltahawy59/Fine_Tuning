import os
import fitz

def run(folder_path):
    
    full_text =""
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            print(f"📄 Processing: {file_name}")
            
            doc = fitz.open(pdf_path)
            full_text += "\n".join(page.get_text() for page in doc)
            print(f"✅ Extracted from {file_name}: {len(full_text)} characters.")
            
            
    
    return full_text
