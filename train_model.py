import torch
import pandas
import os
from datasets import Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq, Trainer
from accelerate import Accelerator
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict


model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# decoder architecture for llama, thus use left padding
tokenizer.padding_side = "left"
tokenizer.pad_token = "[PAD]"


model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=False)

# not needed for fine tuning, want to use updated weights
model.config.use_cache = False



# create pandas dataframes. load preprocessed datasets
dataset_val = load_from_disk("val22")
dataset_train = load_from_disk("train22")



# hyperparameters need to be tested / messed around with
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    # rank 16 for matrices
    r=16,
    # approximate double r for lora_alpha
    lora_alpha=32,
    lora_dropout=0.05,
    # train on all linear layers, not just attention layers
    # note currently using attention layers to reduce overhead / computational cost
     target_modules=["q_proj", "v_proj"])

# load peft configured model
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()

model_dir = f"LLM-13b-170book"
log_dir = f"LLM-13b-170book/logs"


# set up training arguments
training_args = TrainingArguments(
    # output directory
    output_dir = model_dir,
    learning_rate=1e-3,
    # low batch size else CUDA out of memory error
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    num_train_epochs=1,

    # weight decay to avoid overfitting
    weight_decay=0.01,


    # logging strategy
    logging_dir=log_dir,
    logging_strategy="steps",
    logging_steps= 100,

    # number of checkpoints to save (last 20)
    save_total_limit=20,

    # take the best model at the end
    load_best_model_at_end=True,

    # perform evaluation alongside training with steps
    eval_strategy="steps",
)


# takes care of padding for batches
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=peft_model,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    args=training_args,
    data_collator=data_collator
)

# train the model
trainer.train()



trainer.save_model(model_dir)