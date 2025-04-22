# this file is used to preprocess training data

import torch
import pandas
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

path = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(path)

bnb_config = BitsAndBytesConfig(
        # 4 bit quantization
        load_in_4bit = True,
        # nested quantization
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb_config)

# get max input length for the mdoel
def get_max_length(model):
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            break
    if not max_length:
        max_length = 1024
    return max_length

max_length = get_max_length(model)

# create pandas dataframes. change names of excel sheets as necessary
dataset_train = pandas.read_excel("170books_tr.xlsx")
dataset_val = pandas.read_excel("170books_vd.xlsx")

# convert into Huggingface datasets
dataset_train = Dataset.from_pandas(dataset_train)
dataset_val = Dataset.from_pandas(dataset_val)


# template training/validation data
def template_input_train(input_output):

    input =  [{"role": "user", "content": ""},
    {"role": "assistant", "content": input_output["Output"]},]

    return  {"formatted": "tokenizer.apply_chat_template(input, tokenize=False)"}

# template each row in each dataset, and remove standard columns. formatted dict attribute stores the formatted promtp

dataset_train = dataset_train.map(template_input_train, remove_columns=["Output", "Input"])
dataset_val = dataset_val.map(template_input_train, remove_columns=["Output", "Input"])


# function to tokenize batches of prompt formatted input
def tokenize_function(batch):

  # no padding, will pad via data collator dynamically during training
  return tokenizer(batch["formatted"], truncation=True, max_length=max_length, padding=False)

dataset_train = dataset_train.map(tokenize_function, remove_columns=["formatted"])
dataset_val = dataset_val.map(tokenize_function, remove_columns=["formatted"])

print(dataset_train)
print(dataset_val)

# save to disk, change names as necessary
dataset_train.save_to_disk("train170")
dataset_val.save_to_disk("val170")





