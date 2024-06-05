from math import inf
import os
import json
import time
import sys
from time import sleep

from zhipuai import ZhipuAI
from datasets import load_from_disk
from tqdm import tqdm

datasets = ["nq_open", "trivia_qa", "hotpot_qa", "truthful_qa_gen"]
dataset_paths = {
    "nq_open": "/mnt/250T_ceph/smarterliu/llama_eval/lm-evaluation-harness/datasets/nq_open_with_snippets",
    "trivia_qa": "/mnt/250T_ceph/smarterliu/llama_eval/lm-evaluation-harness/datasets/trivia_qa_with_snippets",
}

system_prompt = """
你是一个数据标注专家，你将会被提供一个问题、问题对应的答案以及一些可能相关的文档片段。你需要判断这些文档片段是否包含了问题的答案。文档片段的数量是不定的，最多有20个片段。你需要把所有包含答案的文档片段都标注出来。

数据的格式如下：
{"query": "问题", "label": "答案", "snippets": ["文档片段1", "文档片段2", ...]}

你的输出应该是一个列表，列表的长度与文档片段的数量相同，每个元素是0或者1。0表示对应的文档片段不包含答案，1表示对应的文档片段包含答案。例如，如果有3个文档片段，第1个和第3个包含答案，第2个不包含答案，那么你的输出应该是[1, 0, 1]。
"""

def get_dataset_info(dataset_path, dataset_name="nq_open"):
    """
    The input should be the path to the dataset and the name of dataset.
    The output would be a list of dictionaries, which contain query/label/snippets.
    """
    assert dataset_name in datasets, f"Dataset name {dataset_name} is not supported."
    
    dataset = load_from_disk(dataset_path)
    dataset = dataset["train"]
    res = []
    for data in tqdm(dataset, desc=f"Loading {dataset_name} dataset"):
        if dataset_name == "nq_open":
            query = data["question"]
            label = data["answer"]
            snippets = data["docs"]
            res.append({"query": query, "label": label, "snippets": snippets})
            # if len(res) == 6000:
            #     break
        elif dataset_name == "trivia_qa":
            query = data["question"]
            label = data["answer"]["aliases"][0]
            snippets = data["docs"]
            res.append({"query": query, "label": label, "snippets": snippets})
    
    return res

def form_batch_file(dataset_info, file_base_path, dataset_name="nq_open"):
    """
    dataset_info should be a list of dictionaries, which contain query/label/snippets.
    file_base_path should be the path to save the batch data.
    dataset_name should be the name of dataset.
    """
    assert dataset_name in datasets, f"Dataset name {dataset_name} is not supported."
    
    file_num = -1
    data_index = 0
    file_paths = set()
    for line in tqdm(dataset_info, desc=f"Forming {dataset_name} batch file"):
        ## Limit the number of data in each file to 50000
        if data_index % 50000 == 0:
            file_num += 1
            data_index = 0
        file_path = os.path.join(file_base_path, f"{dataset_name}_batch_{file_num}.jsonl")
        ## Limit the file size to 90MB
        if os.path.exists(file_path) and os.stat(file_path).st_size > 9e7:
            file_num += 1
            data_index = 0
            file_path = os.path.join(file_base_path, f"{dataset_name}_batch_{file_num}.jsonl")
        file_paths.add(file_path)
        custom_id = f"{dataset_name}_{file_num}_{data_index}"
        query = line["query"]
        label = line["label"]
        snippets = line["snippets"]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps({"query": query, "label": label, "snippets": snippets})}
        ]
        line_info = {
            "custom_id": custom_id,  #每个请求必须包含custom_id且是唯一的,长度必须为 6 -64 位.用来将结果和输入进行匹配.
            "method": "POST",
            "url": "/v4/chat/completions", 
            "body": {
                "model": "glm-4",     #每个batch文件只能包含对单个模型的请求,支持 glm-4、glm-3-turbo.
                "messages": messages,
                "temperature": 0.1
            }
        }
        
        with open(file_path, "a") as f:
            f.write(json.dumps(line_info, ensure_ascii=False) + "\n")
        data_index += 1
    
    return list(file_paths)

def monitor_process():
    info_file = "info_file.json"
    with open(info_file, "r") as f:
        info = json.load(f)
    while True:
        status = ""
        for api_key in info.keys():
            client = ZhipuAI(api_key=api_key)
            for batch_id in info[api_key].keys():
                id_status = client.batches.retrieve(batch_id).status
                input_file_id = client.batches.retrieve(batch_id).input_file_id
                output_file_id = client.batches.retrieve(batch_id).output_file_id
                error_file_id = client.batches.retrieve(batch_id).error_file_id
                
                info[api_key][batch_id]["status"] = id_status
                info[api_key][batch_id]["input_file_id"] = input_file_id
                info[api_key][batch_id]["output_file_id"] = output_file_id
                info[api_key][batch_id]["error_file_id"] = error_file_id
                
                if id_status != "completed":
                    status += f"\nbatch id {batch_id} status: {id_status}"
        with open(info_file, "w") as f:
            json.dump(info, f, indent=4)
        if status == "":
            print("All batch jobs are completed.")
            break
        local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        status = f"Update time: {local_time}.\nInfo status: {status}"
        for s in status.split("\n"):
            sys.stdout.write(s + "\n")
            sys.stdout.flush()
        # print(f"Update time: {time}.\nInfo status: {status}")
        sleep(5)
        for _ in range(len(status.split("\n"))):
            sys.stdout.write("\x1b[F\x1b[2K")
    
        
def main():
    file_base_path = "/mnt/250T_ceph/smarterliu/cot_train/data/origin_data/"
    files = ["/mnt/250T_ceph/smarterliu/cot_train/data/origin_data/nq_open/nq_open_error_data.jsonl", "/mnt/250T_ceph/smarterliu/cot_train/data/origin_data/trivia_qa/trivia_qa_error_data.jsonl"]
    client = ZhipuAI(api_key="f9e83de82d1116883bd9634407f77718.rL8FPpPvYJDc0bdp") # smart_liu_18623342873
    # client = ZhipuAI(api_key="7a3c73225504a195beab12b6871a178b.jN5pIovCGncfQiET") # smart_liu_13983347789
    # client = ZhipuAI(api_key="e3de7e608a52d89237b3311de5f49267.zY7eF98HgSrfJXZC") # smart_liu_17623427144
    # client = ZhipuAI(api_key="5c42db10816cc02f1400cab0bdbb5d40.RNoQ2UuDkdPc61Nr") # hebin_1
    # client = ZhipuAI(api_key="70005cc1f3f4a8b0eea223951275d58a.o3qUkPv4ElbbmlXX") # hebin_2
    # client = ZhipuAI(api_key="d9149f31008429e2ce8922ec22546af5.xZQZfnWd4p4asOIJ") # wrh
    
    ## Form batch files for different datasets
    # files = []
    # use_dataset = ["trivia_qa"]
    # for dataset_name in use_dataset:
    #     print(f"Form batch files for {dataset_name}")
    #     dataset_info = get_dataset_info(dataset_paths[dataset_name], dataset_name)
    #     files.extend(form_batch_file(dataset_info, file_base_path, dataset_name))
        
    ## Upload batch files
    job_ids = []
    for file in files:
        print(f"Upload {file} for batch jobs")
        result = client.files.create(
            file=open(file, "rb"),
            purpose="batch"
        )
        job_ids.append(result.id)

    ## Start batch processing
    info_file = "info_file.json"
    with open(info_file, "r") as f:
        info = json.load(f)
    for job_id in job_ids:
        print(f"Start batch job for: {job_id}")
        create = client.batches.create(
            input_file_id=job_id,
            endpoint="/v4/chat/completions", 
            completion_window="24h", #完成时间只支持 24 小时
            metadata={
                "description": f"Information labeling of job {job_id}"
            }
        )
        info[client.api_key][create.id] = {
            "status": create.status,
            "input_file_id": create.input_file_id,
            "output_file_id": create.output_file_id,
            "error_file_id": create.error_file_id
        }
        with open(info_file, "w") as f:
            json.dump(info, f, indent=4)
        
    ## Monitor batch processing
    monitor_process()
        
if __name__ == "__main__":
    # main()
    monitor_process()
