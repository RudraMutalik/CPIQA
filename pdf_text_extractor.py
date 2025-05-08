import os
from threading import Thread

import pymupdf4llm
import re
import pathlib
import sys

skip = []

pdf_path = "cpiqa/pdfs/"

open('errors.txt', 'w').close()

def extract_text(filename):
    out_filename = filename[:-4] + ".txt"

    # if it already exists, continue
    if os.path.isfile("cpiqa/train_val_test_extracted_paragraphs/" + out_filename):
        print(f'File already exists: {out_filename}')
        return
    if filename in skip:
        print(f'File skipped: {filename}')
        return
    try:
        #text = pymupdfllm_with_timeout(pdf_path + filename)
        text = pymupdf4llm.to_markdown(pdf_path + filename)
    except:
        print("Error processing: " + filename)
        with open("errors.txt", "a") as myfile:
            myfile.write(filename + "\n")
        return

    text = text.replace('[o]', '°')  # Use degree symbol
    text = re.sub(r'.*�.*\n', r'\n\n', text)  # Remove replacement lines
    text = re.sub(r'\|.+\|\n', '', text)  # Remove tables
    text = re.sub(r'\n\n(.+\n){,2}(.+[\d\W]\n){2,}', '', text)  # Remove unofficial tables
    text = re.sub(r'\n(Fig\.|Table) \d+\..+(\n.+)*', '', text)  # Remove captions
    text = text.replace('-----', '')
    text = re.sub(r'(\n.+)*(.*((doi)|(DOI)):.*)(\n.+)*', '', text)  # Remove references with "doi:"
    text = re.sub(r'(([A-Z]’)?[A-Z]\w+, [A-Z].).+(\n.+)*', '', text)  # Remove references starting with a name
    text = re.sub(r'#+.*\n', '', text)  # Remove lines starting with '#'
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\n\n(.+\n\n)+', '\n\n', text)  # Remove single lines

    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Replace newlines with spaces paragraph

    if text.count('\n') < 4:
        with open("errors.txt", "a") as myfile:
            print("Error processing: " + filename)
            myfile.write(filename + "\n")
        return

    pathlib.Path("cpiqa/train_val_test_extracted_paragraphs/" + out_filename).write_bytes(text.encode())
    return


directory = os.fsencode(pdf_path)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    extract_text(filename)
