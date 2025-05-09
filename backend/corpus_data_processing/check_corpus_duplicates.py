import os
import time
from collections import defaultdict
from docx import Document
from win32com import client

def convert_doc_to_pdf(doc_path, pdf_path, retry=2):
    if not os.path.exists(doc_path):
        print(f"File not found: {doc_path}")
        return False
    
    doc_path = os.path.abspath(doc_path)  # Ensure absolute path
    pdf_path = os.path.abspath(pdf_path)
    
    for attempt in range(retry):
        try:
            word = client.Dispatch("Word.Application")
            word.Visible = False  # Run in the background
            doc = word.Documents.Open(doc_path)
            doc.SaveAs(pdf_path, FileFormat=17)  # 17 = wdFormatPDF
            doc.Close(False)
            word.Quit()
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {doc_path}: {e}")
            time.sleep(2)  # Wait before retrying
    
    print(f"Final failure: Could not convert {doc_path}")
    return False

def find_and_convert_docs(directory):
    file_map = defaultdict(list)
    
    for root, _, files in os.walk(directory):
        for file in files:
            name, ext = os.path.splitext(file)
            file_map[name.lower()].append((ext.lower(), os.path.join(root, file)))
    
    pdfs = []
    docs = []
    
    for name, paths in file_map.items():
        for ext, path in paths:
            if ext == ".pdf":
                pdfs.append(path)
            elif ext in [".doc", ".docx"]:
                docs.append(path)
    
    print("Documents in the directory:")
    
    if docs:
        print("\nDOC/DOCX files:")
        for doc in docs:
            print(f"  {doc}")
    else:
        print("\nNo DOC/DOCX files found.")
    
    if pdfs:
        print("\nPDF files:")
        for pdf in pdfs:
            print(f"  {pdf}")
    else:
        print("\nNo PDF files found.")
    
    for doc_path in docs:
        name, ext = os.path.splitext(doc_path)
        pdf_path = name + ".pdf"
        
        if os.path.exists(pdf_path):
            print(f"PDF already exists, skipping: {pdf_path}")
            continue
        
        if ext == ".docx":
            try:
                doc = Document(doc_path)
                doc.save(pdf_path)
                os.remove(doc_path)
                print(f"Converted and removed: {doc_path}")
            except Exception as e:
                print(f"Failed to convert {doc_path}: {e}")
        elif ext == ".doc":
            success = convert_doc_to_pdf(doc_path, pdf_path)
            if success:
                os.remove(doc_path)
                print(f"Converted and removed: {doc_path}")
            else:
                print(f"Failed to convert: {doc_path}")
    
    duplicates = {name: paths for name, paths in file_map.items() if len(paths) > 1}
    
    if duplicates:
        print("\nDuplicate documents found:")
        for name, paths in duplicates.items():
            print(f"\n{name}:")
            for ext, path in paths:
                print(f"  {path}")
    else:
        print("\nNo duplicate documents found.")

if __name__ == "__main__":
    directory = os.path.abspath("../../document_corpus")  # Absolute path
    if os.path.isdir(directory):
        find_and_convert_docs(directory)
    else:
        print("Invalid directory path.")
