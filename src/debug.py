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

def check_gemma_dataset_format():
    tokenizer = AutoTokenizer.from_pretrained("/mnt/250T_ceph/smarterliu/models/Gemma-7B")
    dataset = load_from_disk("/mnt/250T_ceph/smarterliu/llama_eval/lm-evaluation-harness/datasets/alpaca")
    formatting_func = formatting_constant_length_func
    seq_length = 1600
    eos_token_id = tokenizer.eos_token_id
    dataset = ConstantLengthDataset(
        tokenizer=tokenizer,
        dataset=dataset,
        formatting_func=formatting_func,
        seq_length=seq_length,
        eos_token_id=eos_token_id,
    )
    return dataset

