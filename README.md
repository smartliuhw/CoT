# Intro

This repo is used for @smartliuhw thesis's model training. The [huggingface SFT trainer](https://huggingface.co/docs/trl/sft_trainer) is used as the training framwork with deepspeed methodology to ensure the RTX 4090 GPU can be used properly.

# How to use

## Install the dependency

The environment dependency is listed in the [requirment file](./requirements.txt), just run the following command:

```bash
pip install -r requirements.txt
```

## Modify data process file

The data processing code is in the [utils.py](./src/utils.py) file, all the data should be stored with the ``Dataset`` class. The function ``get_train_data`` is the most important part, modify it accroding to your demand.

## Modify train file

The model tran code is in the [train.py](./src/train.py) file, using [trl](https://huggingface.co/docs/trl/sft_trainer) framework. Modify the args, templates, special tokens accroding to your demand.

## Run trainning

You can launch a training by following the [train example](./scripts/train_example.sh) file, with only few changes about the model and the data.

If you have any question, feel free to ask me.
