# models/llm/train_llm.py
# This script fine-tunes a GPT-2 model on your SMILES dataset using Hugging Face's Transformers library.

import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def train_language_model():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Prepare dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="data/processed/cleaned_smiles.csv",  # Ensure this file contains SMILES strings
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir="models/llm/output",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10,
        save_total_limit=2,
        logging_steps=5
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    trainer.train()
    trainer.save_model("models/llm/output")
    tokenizer.save_pretrained("models/llm/output")
    print("[✔] LLM model trained and saved.")
