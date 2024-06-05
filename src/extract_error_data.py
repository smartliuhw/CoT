import os
import json

from tqdm import tqdm

datasets = ["nq_open", "trivia_qa", "hotpot_qa", "truthful_qa_gen"]

def get_whole_data(origin_data_folder, data_dict=None):
    """
    The data folder should contain original json or jsonl files.
    The data_dict should be a dictionary with custom_id as key and data as value. If not provided, it will be an empty dictionary.
    """
    if data_dict is None:
        data_dict = {}
    for file in tqdm(os.listdir(origin_data_folder), desc="Loading origin data"):
        if (file.endswith("json") or file.endswith("jsonl")) and "error" not in file:
            with open(os.path.join(origin_data_folder, file), "r") as f:
                lines = f.readlines()
                lines = [json.loads(line) for line in lines]
            for line in lines:
                data_dict[line["custom_id"]] = line
    return data_dict

def get_error_data(data_dict, dataset_name, origin_data_folder, out_file_folder):
    """
    The data_dict should be a dictionary with custom_id as key and data as value.
    The dataset_name should be the name of dataset.
    The origin_data_folder should be the path to the original data.
    The out_file_folder should be the path to save the error data.
    """
    assert dataset_name in datasets, f"Dataset name {dataset_name} is not supported."
    assert data_dict is not None, "The data_dict should not be None."
    error_data = []
    for file in tqdm(os.listdir(out_file_folder), desc="Checking error data"):
        if "error" in file and dataset_name in file:
            with open(os.path.join(out_file_folder, file), "r") as f:
                lines = f.readlines()
                lines = [json.loads(line) for line in lines]
            error_data.extend(data_dict[line["custom_id"]] for line in lines)
            
    with open(os.path.join(origin_data_folder, f"{dataset_name}_error_data.jsonl"), "w") as f:
        for data in error_data:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    
    print(f"Error data for {dataset_name} has been saved to {origin_data_folder}.")
    
def main():
    origin_base_data_folder = "/mnt/250T_ceph/smarterliu/cot_train/data/origin_data"
    out_file_folder = "/mnt/250T_ceph/smarterliu/cot_train/data/glm_response"
    parse_dataset = ["nq_open", "trivia_qa"]
    for dataset_name in parse_dataset:
        data_dict = get_whole_data(os.path.join(origin_base_data_folder, dataset_name), data_dict=None)
        get_error_data(data_dict, dataset_name, os.path.join(origin_base_data_folder, dataset_name), out_file_folder)
        
if __name__ == "__main__":
    main()
