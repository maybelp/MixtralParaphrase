
import os
import gc
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction

import torch
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from huggingface_hub import HfFolder
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.nn import CrossEntropyLoss
from evaluate import load


# ------------------ Setup & Configuration ------------------ #
def setup_environment():
    print("\n✅ PyTorch is installed.")
    print("🔢 Version:", torch.__version__)
    print("🚀 CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("🖥️ CUDA device name:", torch.cuda.get_device_name(0))

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        HfFolder.save_token(hf_token)
        print("✅ Hugging Face token saved.")
    else:
        print("❌ HF_TOKEN environment variable not found.")


# ------------------ Paths ------------------ #
MIXTRAL_DATA_DIR = os.getenv("MIXTRAL_DATA", "/home/hmc/lmp42/mixtral_data")
PARAPHRASED_DIR = os.path.join(MIXTRAL_DATA_DIR, "paraphrased_tokenized_train")
COMBINED_DIR = os.path.join(MIXTRAL_DATA_DIR, "combined_tokenized_train")
OFFLOAD_DIR = os.path.join(MIXTRAL_DATA_DIR, "offload")
os.makedirs(OFFLOAD_DIR, exist_ok=True)


# ------------------ Load Paraphrasing Model ------------------ #
def load_paraphrasing_model():
    model_name = "mrm8488/bert2bert_shared-spanish-finetuned-paus-x-paraphrasing"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)


# ------------------ Dataset Utilities ------------------ #
def load_translation_dataset():
    urls = {
        "train": "https://huggingface.co/datasets/somosnlp-hackathon-2022/spanish-to-quechua/resolve/main/data/train-00000-of-00001.parquet"
    }
    return load_dataset("parquet", data_files=urls["train"])["train"]


def paraphrase_spanish_phrase(generator, phrase, k=3, max_new_tokens=100):
    paraphrases = []
    for _ in range(k):
        prompt = f"""<s>[INST] Parafrasea esta frase en español manteniendo el mismo significado: \"{phrase}\" [/INST]"""
        output = generator(prompt, max_new_tokens=max_new_tokens, do_sample=True, top_p=0.9, temperature=0.8)[0]["generated_text"]
        try:
            response = output.split("[/INST]")[-1].strip()
            paraphrases.append(response)
        except IndexError:
            paraphrases.append(output.strip())
    return paraphrases


def generate_augmented_dataset(generator, dataset, k=1, batch_size=8):
    sources, targets = [], []
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating paraphrases"):
        batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
        for row in batch:
            es, qu = row["es"], row["qu"]
            try:
                para_es_list = paraphrase_spanish_phrase(generator, es, k=k)
            except Exception:
                para_es_list = [es] * k
            for p in para_es_list:
                sources.append(p)
                targets.append(qu)
    return Dataset.from_dict({"source": sources, "target": targets})


# ------------------ Tokenization ------------------ #
def format_prompt(example):
    return {
        "text": f"""Traduce del español al quechua:\n{example['source']}\nRespuesta:""",
        "labels": example["target"]
    }


def tokenize_for_mixtral(example):
    mixtral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    full_input = f"{example['text']} {example['labels']}"
    tokens = mixtral_tokenizer(full_input, truncation=True, padding="max_length")
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


# ------------------ Main ------------------ #
def main():
    setup_environment()

    generator = load_paraphrasing_model()
    raw_train = load_translation_dataset()

    # Generate paraphrased data
    augmented_dataset = generate_augmented_dataset(generator, raw_train, k=1)

    # Format and tokenize
    formatted = augmented_dataset.map(format_prompt)
    tokenized_augmented = formatted.map(tokenize_for_mixtral)

    # Final cleanup
    keep_cols = ["input_ids", "attention_mask", "labels"]
    tokenized_augmented = tokenized_augmented.remove_columns(
        [col for col in tokenized_augmented.column_names if col not in keep_cols]
    )
    tokenized_augmented.set_format(type="torch", columns=keep_cols)

    # Save paraphrased dataset
    tokenized_augmented.save_to_disk(PARAPHRASED_DIR)
    print(f"\n📁 Paraphrased dataset saved to: {PARAPHRASED_DIR}")

    # Combine with original dataset
    original_path = os.path.join(MIXTRAL_DATA_DIR, "tokenized_train")
    original_dataset = load_from_disk(original_path)
    combined_dataset = concatenate_datasets([original_dataset, tokenized_augmented])
    combined_dataset.save_to_disk(COMBINED_DIR)
    print(f"✅ Combined dataset saved to: {COMBINED_DIR}")





if __name__ == "__main__":
    main()
