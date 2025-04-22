import torch
import transformers
import pandas
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import trange, tqdm
from transformers.pipelines.pt_utils import KeyDataset




model_id = "meta-llama/Llama-2-13b-chat-hf"

# load the preprocessed test dataset.
dataset_test = load_from_disk("testprompts")

model = AutoModelForCausalLM.from_pretrained(model_id)
model = PeftModel.from_pretrained(model, f"LLM-13b-88book")
model = model.merge_and_unload()
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer = tokenizer,
    torch_dtype=torch.float16,
    device=0,
)

# generate responses
storage_list = []
for prompt_sample in dataset_test:

    prediction = pipeline(prompt_sample["formatted"],  num_return_sequences=1,
                           max_new_tokens=128)[0]['generated_text'][-1]
    storage_list.append({"inference": prediction["content"], "answer": prompt_sample["answer"]})
    print(prediction["content"], flush=True)

# convert responses into an excel spreadsheet, change the name of the spreadsheet as necessary/
df = pandas.DataFrame.from_records(storage_list)
print(df)
df.to_excel("testanswers-88book-7b.xlsx")
    
    
