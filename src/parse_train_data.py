import os
import json
import random
import re

from tqdm import tqdm

from extract_error_data import datasets, get_whole_data

def parse_response(response, custom_id):
    list_pattern = re.compile(r'\[(.*?)\]')
    match = list_pattern.search(response)
    
    try:
        # 提取列表部分并去除空格和换行符
        list_str = match.group(1).replace('\n', '').replace(' ', '')
        # 将字符串转化为列表
        result_list = [int(x) for x in list_str.split(',') if x in '01']
        
        # 提取备注部分
        remark = response.replace(match.group(0), '').strip()
    except:
        print(f"Error in parsing response for {custom_id}.")
        result_list = []
        remark = response.strip()
    
    return result_list, remark
    

def parse_glm_out(out_file_folder, dataset_name, origin_data_folder, out_folder):
    """
    The out_file_folder should be the path to the glm response.
    The dataset_name should be the name of dataset.
    The origin_data_folder should be the path to the original data.
    The out_folder should be the path to save the parsed data.
    """
    assert dataset_name in datasets, f"Dataset name {dataset_name} is not supported."
    data_dict = get_whole_data(origin_data_folder, data_dict=None)
    parsed_data = []
    for file in tqdm(os.listdir(out_file_folder), desc="Parsing glm response"):
        if "res" in file and dataset_name in file:
            with open(os.path.join(out_file_folder, file), "r") as f:
                lines = f.readlines()
                lines = [json.loads(line) for line in lines]
            
            for line in lines:
                custom_id = line["custom_id"]
                response = line["response"]["body"]["choices"][0]["message"]["content"]
                result_list, remark = parse_response(response, custom_id)
                
                origin_data = data_dict[custom_id]
                content = origin_data["body"]["messages"][1]["content"]
                content = json.loads(content)
                query, label, snippets = content["query"], content["label"], content["snippets"]
                
                parsed_data.append(
                    {
                        "custom_id": custom_id,
                        "query": query,
                        "label": label,
                        "snippets": snippets,
                        "result": result_list,
                        "remark": remark
                    }
                )
                
                
    random.shuffle(parsed_data)
    with open(os.path.join(out_folder, f"{dataset_name}_parsed_data.jsonl"), "w") as f:
        for data in parsed_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    print(f"Parsed data for {dataset_name} has been saved to {out_folder}.")


def main():
    origin_base_data_folder = "/mnt/250T_ceph/smarterliu/cot_train/data/origin_data"
    out_file_folder = "/mnt/250T_ceph/smarterliu/cot_train/data/glm_response"
    train_data_folder = "/mnt/250T_ceph/smarterliu/cot_train/data/train_data"
    parse_dataset = ["nq_open", "trivia_qa"]
    for dataset_name in parse_dataset:
        parse_glm_out(out_file_folder, dataset_name, os.path.join(origin_base_data_folder, dataset_name), train_data_folder)
        
if __name__ == "__main__":
    main()
    