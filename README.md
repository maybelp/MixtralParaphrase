# Mixtral Paraphrasing Data Augmentation Pipeline

This project uses a Spanish-to-Quechua dataset and applies data augmentation via Spanish paraphrasing. The goal is to improve translation performance by enriching the training set using paraphrased prompts.

---
> **Paraphrasing Augmentation for Quechua–Spanish Translation**  
> Using  Transformers and BERT2BERT Spanish paraphraser  
> Author: Ligia Palomo

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model](https://img.shields.io/badge/model-BERT2BERT%20PaUS-green)](https://huggingface.co/mrm8488/bert2bert_shared-spanish-finetuned-paus-x-paraphrasing)

---
##  Project Structure
'''
MixtralParaphrase/
├── data/
├── results/
├── src/
│   ├── MixtralParaphrasePipeline/
│   │   ├── __init__.py
│   │   └── paraphraser.py      ✅ core logic here
│   └── run_paraphraser.py      ✅ minimal entrypoint
├── tests/
│   └── test_paraphraser.py     ✅ test lives outside src/
├── .gitignore
├── README.md
├── requirements.txt

'''

---

##  Environment Setup

### 1. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate

##  Install dependencies
'''
pip install -r requirements.txt

'''
## Usage
'''
python src/run_paraphraser.py
'''
This will:

Load the original Spanish–Quechua dataset

Use BERT2BERT to generate paraphrases for the Spanish source

Format the data with prompts suitable for Mixtral

Tokenize using mistralai/Mixtral-8x7B-Instruct-v0.1

Save the processed data to disk

## Output
paraphrased_tokenized_train/: Augmented paraphrased dataset

combined_tokenized_train/: Merged with original tokenized dataset

## Dependencies
Transformers
Datasets
Evaluate
Torch
NLTK
TQDM
Matplotlib
Pandas
Author
Ligia Palomo



