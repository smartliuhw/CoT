import os
import sys
import random
import json
import argparse
from dataclasses import dataclass, field

import torch
from tqdm import tqdm
import pandas as pd
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from trl.trainer import ConstantLengthDataset
from datasets import load_from_disk, concatenate_datasets, Dataset, DatasetDict
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser

from utils import get_train_data, formatting_prompts_func, formatting_constant_length_func, model_name_to_path

@dataclass
class ModelArguments:
    model_type: str = field(default=None, metadata={"help": "The model name."})
    
@dataclass
class DataArguments:
    train_data: str = field(default=None, metadata={"help": "Choose the training data, split by comma."})
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    max_seq_length: int = field(default=8192, metadata={"help": "The cache directory."})

def train():
    # args = parse_args()
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set random seed
    random.seed(training_args.seed)

    # Load model and tokenizer
    model_path = model_name_to_path[model_args.model_type]
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    # Load training data
    print("Loading training data...")
    train_data = get_train_data(data_args.train_data, training_args.seed)
    train_data = ConstantLengthDataset(
            tokenizer,
            train_data,
            formatting_func=formatting_constant_length_func,
            seq_length=training_args.max_seq_length,
            num_of_sequences=100,
    )
    # print(train_data)

    # Load trainer
    trainer = SFTTrainer(
        model,
        training_args,
        train_dataset=train_data,
        # dataset_text_field="text",
        # formatting_func=formatting_prompts_func,
        packing=True,
        max_seq_length=training_args.max_seq_length,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(training_args.output_dir)
    
if __name__ == "__main__":
    train()
