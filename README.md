# Intro

This repo is used for @smartliuhw thesis's model training. The [huggingface SFT trainer](https://huggingface.co/docs/trl/sft_trainer) is used as the training framwork with deepspeed methodology to ensure the RTX 4090 GPU can be used properly.

# How to use

The environment dependency is listed in the [requirment file](./requirements.txt), just run the following command:

```bash
pip install -r requirements.txt
```

All the source code files are in the [src](./src) folder, and you can launch a training by following the [train example](./scripts/train_example.sh) file, with only few changes about the model and the data.

If you have any question, feel free to ask me.
