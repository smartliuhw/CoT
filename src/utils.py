import os
import sys
import random
import json
import argparse
from dataclasses import dataclass, field

from tqdm import tqdm
import pandas as pd
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_from_disk, concatenate_datasets, Dataset, DatasetDict
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser

avail_datasets = [
    "universal_instruct",
    "nq_open",
    "nq_open_snippets_label",
    "trivia_qa",
    "trivia_qa_snippets_label",
    "hotpot_qa",
    "hotpot_qa_snippets_label",
    "nq_open_with_snippets",
    "trivia_qa_with_snippets",
    "hotpot_qa_with_snippets",
    "nq_open_cot",
    "trivia_qa_cot",
    "hotpot_qa_cot",
]

dataset_name_to_path = {
    "universal_instruct": "/home/shared_space/smart/data/alpaca",
    "nq_open": "/home/shared_space/smart/data/nq_open",
    "trivia_qa": "/home/shared_space/smart/data/trivia_qa",
    "hotpot_qa": "/home/shared_space/smart/data/hotpot_qa",
    "hotpot_qa_snippets_label": "/home/shared_space/smart/data/hotpot_qa",
    "nq_open_snippets_label": "../data/train_data/nq_open_parsed_data.jsonl",
    "trivia_qa_snippets_label": "../data/train_data/trivia_qa_parsed_data.jsonl",
    "nq_open_with_snippets": "../data/train_data/nq_open_parsed_data.jsonl",
    "trivia_qa_with_snippets": "../data/train_data/trivia_qa_parsed_data.jsonl",
    "hotpot_qa_with_snippets": "/home/shared_space/smart/data/hotpot_qa",
    "nq_open_cot": "../data/train_data/nq_open_parsed_data.jsonl",
    "trivia_qa_cot": "../data/train_data/trivia_qa_parsed_data.jsonl",
    "hotpot_qa_cot": "/home/shared_space/smart/data/hotpot_qa",
}

model_name_to_path = {
    "Gemma-2B": "/home/shared_space/smart/models/Gemma-2B",
    "Gemma-7B": "/home/shared_space/smart/models/Gemma-7B",
    "Llama2-7B": "/home/shared_space/smart/models/Llama2-7B",
    "Llama3-8B": "/home/shared_space/smart/models/Llama3-8B",
    "Mistral-7B": "/home/shared_space/smart/models/Mistral-7B",
}

model_name_to_sep_token = {
    "Gemma-2B": "<unused2>",
    "Gemma-7B": "<unused2>",
    "Llama2-7B": "<0x02>",
    "Llama3-8B": "<|reserved_special_token_2|>",
    "Mistral-7B": "<0x02>",
}

def add_instruction(batch):
    # example["instruction"] = "Answer the following question."
    return {"instruction": ["Answer the following question." for _ in batch["question"]]}

def load_jsonl_to_dataset(jsonl_file):
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        desc = f"Loading {jsonl_file.split('/')[-1]}"
        for line in tqdm(f, desc=desc):
            line = json.loads(line)
            query = line["query"]
            snippets = line["snippets"]
            labels = line["result"]
            if len(snippets) != len(labels):
                continue
            instruction = "You are given the following question and snippets. Please select the snippets that are relevant to the question.\nYour output should be the sequence number of the selected snippets. Here is an output example:\n[1, 2, 4]\nThis means the first, second, and fourth snippets are selected."
            snippets_input = ""
            for i, snippet in enumerate(snippets):
                snippets_input += f"{i+1}. {snippet}\n"
            input = f"question: {query}\n\nsnippets:\n{snippets_input}"
            output = str([i+1 for i, label in enumerate(labels) if label == 1])
            data.append({"instruction": instruction, "input": input, "output": output})
    return Dataset.from_pandas(pd.DataFrame(data))

def process_nq_open_with_snippets(data):
    processed_data = []
    for item in tqdm(data, desc="Processing nq_open's snippets label"):
        question = item["query"]
        snippets = item["snippets"]
        labels = item["label"]
        facts = ""
        for idx, snippet in enumerate(snippets[:5]):
            facts += f"{idx+1}. {snippet}\n\n"
        input = f"Q: {question}\nA:"
        output = labels[0]
        instruction = f"Answer these questions based on the given facts: {facts.strip()}\n\n"
        processed_data.append({"instruction": instruction, "input": input, "output": output})
    
    return Dataset.from_pandas(pd.DataFrame(processed_data))

def process_trivia_qa_with_snippets(data):
    processed_data = []
    for item in tqdm(data, desc="Processing trivia_qa's snippets label"):
        question = item["query"]
        snippets = item["snippets"]
        label = item["label"]
        facts = ""
        for idx, snippet in enumerate(snippets[:5]):
            facts += f"{idx+1}. {snippet}\n\n"
        input = f"Question: {question}\nAnswer:"
        output = label
        instruction = f"Please use the following facts {facts.strip()} to answer the question"
        processed_data.append({"instruction": instruction, "input": input, "output": output})
    
    return Dataset.from_pandas(pd.DataFrame(processed_data))

def process_hotpot_qa_with_snippets(data):
    processed_data = []
    for item in tqdm(data, desc="Processing hotpot_qa's snippets label"):
        question = item["question"]
        label = item["answer"]
        titles = item["context"]["title"]
        sentences = item["context"]["sentences"]
        title_to_sentences = {}
        for title, sents in zip(titles, sentences):
            title_to_sentences[title] = sents
        facts = ""
        for i in range(min(5, len(item["supporting_facts"]))):
            fact_title = item["supporting_facts"]["title"][i]
            fact_id = item["supporting_facts"]["sent_id"][i]
            fact = title_to_sentences[fact_title][fact_id]
            facts += f"{i + 1}. {fact}"
        input = f"Q:\n{question}\nAnswer:\n"
        output = label
        instruction = f"There are several facts:\n{facts.strip()}\nPlease answer the following question according to these facts."
        processed_data.append({"instruction": instruction, "input": input, "output": output})
    
    return Dataset.from_pandas(pd.DataFrame(processed_data))
        

def process_jsonl_cot(jsonl_file, model_name):
    processed_data = []
    random.seed(725)
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        desc = f"Loading {jsonl_file.split('/')[-1]}"
        for line in tqdm(f, desc=desc):
            line = json.loads(line)
            if line["remark"] != "":
                continue
            question = line["query"]
            snippets = line["snippets"]
            answer = line["label"][0]
            snippet_labels = line["result"]
            
            instruction = """You are given the following question and snippets. You should select the snippets that are relevant to the question first, and answer the question according to these snippets.\nYour output should be to first output the serial number of the selected snippet in the form of a list, and then splice the correct answer at the end."""

            snippets_input = ""
            snippet_index = 1
            output_index = []
            for label, snippet in zip(snippet_labels, snippets)[:5]:
                snippets_input += f"{snippet_index}. {snippet.strip()}\n"
                if label == 1:
                    output_index.append(snippet_index)
                snippet_index += 1
            
            input_text = f"question: {question}\n\nsnippets:\n{snippets_input}"
            output_text = str(output_index) + model_name_to_sep_token[model_name] + answer
            
            processed_data.append({"instruction": instruction, "input": input_text, "output": output_text})
        
    return Dataset.from_pandas(pd.DataFrame(processed_data))


def process_hotpot_qa_cot(data, model_name):
    processed_data = []
    random.seed(725)
    for item in tqdm(data, desc="Processing hotpot_qa's cot label"):
        question = item["question"]
        titles = item["context"]["title"]
        sentences = item["context"]["sentences"]
        supporting_facts_titles = item["supporting_facts"]["title"]
        supporting_facts_sent_ids = item["supporting_facts"]["sent_id"]
        answer = item["answer"]
        
        instruction = """You are given the following question and snippets. You should select the snippets that are relevant to the question first, and answer the question according to these snippets.\nYour output should be to first output the serial number of the selected snippet in the form of a list, and then splice the correct answer at the end."""

        # for idx, (title, sents) in enumerate(zip(titles, sentences)):
        #     for sent_idx, sent in enumerate(sents):
        #         snippets_input += f"{snippet_index}. {sent.strip()}\n"
        #         snippets_list.append((title, sent_idx))
        #         snippet_index += 1
        
        # input_text = f"question: {question}\n\nsnippets:\n{snippets_input}"
        
        # output = []
        # for title, sent_id in zip(supporting_facts_titles, supporting_facts_sent_ids):
        #     for idx, (snippet_title, snippet_sent_id) in enumerate(snippets_list):
        #         if title == snippet_title and sent_id == snippet_sent_id:
        #             output.append(idx + 1)
        
        supporting_dict = {}
        for title, sent_id in zip(supporting_facts_titles, supporting_facts_sent_ids):
            supporting_dict[title] = supporting_dict.get(title, []) + [sent_id]
            
        supporting_facts = []
        unsupported_facts = []
        for idx, (title, sents) in enumerate(zip(titles, sentences)):
            sent_id = supporting_dict.get(title, [])
            for sent_idx, sent in enumerate(sents):
                if sent_idx in sent_id:
                    supporting_facts.append(sent)
                else:
                    unsupported_facts.append(sent)
        
        selected_supporting_facts = random.sample(supporting_facts, min(5, len(supporting_facts)))
        selected_unsupported_facts = random.sample(unsupported_facts, min(5 - len(selected_supporting_facts), len(unsupported_facts)))
        selected_facts = selected_supporting_facts + selected_unsupported_facts
        random.shuffle(selected_facts)
        
        ## assign index to each snippet randomly
        snippets_input = ""
        output_index = []
        for idx, sent in enumerate(selected_facts):
            snippets_input += f"{idx + 1}. {sent.strip()}\n"
            if sent in selected_supporting_facts:
                output_index.append(idx + 1)
        
        input_text = f"question: {question}\n\nsnippets:\n{snippets_input}"
        output_text = str(output_index) + model_name_to_sep_token[model_name] + answer
        
        processed_data.append({"instruction": instruction, "input": input_text, "output": output_text})
    
    return Dataset.from_pandas(pd.DataFrame(processed_data))

def get_train_data(train_data, model_name, seed=725):
    # Split train data name
    datasets_info = train_data.split("|")
    assert all(dataset_info.split("-")[1] in avail_datasets for dataset_info in datasets_info), f"Dataset not supported, choose from {avail_datasets}"
    parsed_dataset = []
    random.seed(seed)
    for dataset_info in datasets_info:
        dataset_num, dataset_name = dataset_info.split("-")
        print(f"Loading {dataset_name}...")
        dataset_path = dataset_name_to_path[dataset_name]
        if dataset_name == "universal_instruct":
            dataset = load_from_disk(dataset_path)["train"]
            dataset = dataset.remove_columns(["text"])
        elif dataset_name == "nq_open":
            dataset = load_from_disk(dataset_path)["train"]
            dataset = dataset.map(add_instruction, batched=True, desc=f"Adding instruction for {dataset_name}")

            def extract_answer(batch):
                # example["answer"] = example["answer"][0]
                return {"answer": [answer[0] for answer in batch["answer"]]}
            dataset = dataset.map(extract_answer, batched=True, desc=f"Extracting answer for {dataset_name}")
            
            dataset = dataset.rename_column("question", "input")
            dataset = dataset.rename_column("answer", "output")
        elif dataset_name == "trivia_qa":
            dataset = load_from_disk(dataset_path)["train"]
            dataset = dataset.map(add_instruction, batched=True, desc=f"Adding instruction for {dataset_name}")
            
            def extract_answer(batch):
                # example["answer"] = example["answer"]["aliases"][0]
                return {"answer": [answer["aliases"][0] for answer in batch["answer"]]}
            dataset = dataset.map(extract_answer, batched=True, desc=f"Extracting answer for {dataset_name}")
            
            dataset = dataset.rename_column("question", "input")
            dataset = dataset.rename_column("answer", "output")
            all_columns = dataset.column_names
            columns_to_remove = [col for col in all_columns if col not in ["instruction", "input", "output"]]
            dataset = dataset.remove_columns(columns_to_remove)
        elif dataset_name == "hotpot_qa":
            dataset = load_from_disk(dataset_path)["train"]
            dataset = dataset.map(add_instruction, batched=True, desc=f"Adding instruction for {dataset_name}")

            dataset = dataset.rename_column("question", "input")
            dataset = dataset.rename_column("answer", "output")
            all_columns = dataset.column_names
            columns_to_remove = [col for col in all_columns if col not in ["instruction", "input", "output"]]
            dataset = dataset.remove_columns(columns_to_remove)
        elif dataset_name == "nq_open_snippets_label" or dataset_name == "trivia_qa_snippets_label":
            dataset = load_jsonl_to_dataset(dataset_path)
        elif dataset_name == "hotpot_qa_cot":
            dataset = load_from_disk(dataset_path)["train"]
            dataset = process_hotpot_qa_cot(dataset)
        elif dataset_name == "nq_open_with_snippets":
            datas = []
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    datas.append(json.loads(line))
            dataset = process_nq_open_with_snippets(datas)
        elif dataset_name == "trivia_qa_with_snippets":
            datas = []
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    datas.append(json.loads(line))
            dataset = process_trivia_qa_with_snippets(datas)
        elif dataset_name == "hotpot_qa_with_snippets":
            dataset = load_from_disk(dataset_path)["train"]
            dataset = process_hotpot_qa_with_snippets(dataset)
        elif dataset_name == "nq_open_cot" or dataset_name == "trivia_qa_cot":
            dataset = process_jsonl_cot(dataset_path, model_name)
        elif dataset_name == "hotpot_qa_cot":
            dataset = load_from_disk(dataset_path)["train"]
            dataset = process_hotpot_qa_cot(dataset, model_name)
        
        dataset = dataset.shuffle(seed=seed)
        selected_dataset = dataset[:min(len(dataset), int(dataset_num))]
        dataset = dataset.from_dict(selected_dataset)
        parsed_dataset.append(dataset)
            
    all_train_datas = concatenate_datasets(parsed_dataset).shuffle(seed=seed)
    return all_train_datas
    # return DatasetDict({"train": all_train_datas})

def formatting_prompts_func(examples):
    output_text = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        input = examples["input"][i]
        output = examples["output"][i]

        if len(input) >= 2:
            text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}'''
        else:
            text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
            
### Instruction:
{instruction}

### Response:
{output}'''
        output_text.append(text)

    return output_text

def formatting_constant_length_func(example):
    instruction = example["instruction"]
    input = example["input"]
    output = example["output"]
    if len(input) >= 2:
        text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
        {output}
        '''
    else:
        text = f'''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Response:
        {output}
        '''
    return text
