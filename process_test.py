# this file is used to template the test data

import pandas
from datasets import Dataset

# change the excel shee. Then run teh code to preprocess test data
dataset_test = pandas.read_excel("TESTSHEET.xlsx")
dataset_test = Dataset.from_pandas(dataset_test)

# template test data. we don't include an answer to the question as we are simply prompting the model
def template_input_test(instruction_answer):
    # modify prompts as necessary
    input =  [{"role": "system", "content": "You are an expert in water and wastewater treatment and engineering. The following is a multiple-choice question about water and wastewater treatment knowledge. State the correct letter choice first and answer concisely."},
    {"role": "user", "content": instruction_answer["Question"] + " Correct Letter of the Answer:"},]
    return {"formatted": input, "answer": instruction_answer["Answer"]}



dataset_test = dataset_test.map(template_input_test, remove_columns=["chapter", "Question", "Answer", "Full_answer", "ind"])

# change name of the saved test dataset as necessary
dataset_test.save_to_disk("testprompts2")

