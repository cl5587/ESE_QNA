from accelerate import Accelerator
from datasets import load_from_disk
from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean
from peft import PeftModel
import pandas
import torch, time, json

accelerator = Accelerator()

dataset_test = load_from_disk("testprompts2")

# load a base model and tokenizer
model_path = "meta-llama/Llama-2-13b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    device_map={"": accelerator.process_index},
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(model, f"LLM-13b-88book")
model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(model_path)   

# sync GPUs and start the timer
accelerator.wait_for_everyone()
start=time.time()

# divide the prompt list onto the available GPUs 
with accelerator.split_between_processes(dataset_test) as prompts:
    # store output of generations in dict
    results=[]

    # have each GPU do inference, prompt by prompt
    for prompt in prompts:
        # get the formatted prompt froM the dataset
        prompt_formatted = tokenizer.apply_chat_template(prompt["formatted"], tokenize=False)
        prompt_tokenized=tokenizer(prompt_formatted, return_tensors="pt").to("cuda")
        output_tokenized = model.generate(**prompt_tokenized, max_new_tokens=100)[0]

        # remove prompt from output 
        output_tokenized=output_tokenized[len(prompt_tokenized["input_ids"][0]):]

        # store outputs and number of tokens in result{}
        final_output = tokenizer.decode(output_tokenized)
        results.append({"inference": final_output, "answer": prompt["answer"]})

  #  results=[ results ] # transform to list, otherwise gather_object() will not collect correctly

# collect results from all the GPUs
results_gathered=gather_object(results)
print(results_gathered)
df = pandas.DataFrame.from_records(results_gathered)
print(df)
df.to_excel("tester.xlsx")

if accelerator.is_main_process:
    timediff=time.time()-start

    print(f"time {timediff}")

   