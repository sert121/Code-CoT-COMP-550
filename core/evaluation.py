from human_eval.data import write_jsonl, read_problems
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)
from tqdm import tqdm
import itertools
import typing
from typing import Union
import json,gzip

BatchGenerator = typing.Callable[
    [PreTrainedModel, PreTrainedTokenizer, str, int], list[str]
]

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        # if not template_name:
        #     # Enforce the default here, so the constructor can be called with '' and will not break.
        #     template_name = "alpaca"

        self.template = {
        "description": "Template used by Alpaca-LoRA.",
        "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
        "prompt_no_input": "[INST] Below is an instruction that describes a task. Write a response that appropriately completes the request:\n{instruction}\n[/INST]",
        "response_split": "[/INST]"    
        }
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
        )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()




def create_mbpp_instruction(example):
    description = example["text"]
    test_cases = example["test_list"]
    prompt = "You are an expert Python programmer, and you need to respond with code to complete is your task: {description} Your code should pass these tests:\n\n{tests}\n." 

    # prompt = "You are an expert Python programmer, and you need to respond with code to complete is your task. Your code should pass certain tests. You shall also be provided with some steps to help you solve the problem.  \n Task: {description}. \n Tests: \n\n{tests}\n."
    # prompt = "Task:\n{description}\nTests:\n{tests}\n." # comment if not using gpt

    #prompt modified for mbpp codellama
    # prompt =  " Remember to respond with only the function definition, nothing else. \nTask: {description} \nTests:\n {tests}\nCode:"
    # prompt =  "You are an expert Python programmer, and you need to respond with code to complete is your task. "


    #promptfor mbpp codellama (zero-shot steps)
    # prompt = " Remember to respond with only the function definition, nothing else. You shall be provided steps to generate more accurate code. \nTask: {description} \nTests:\n {tests}"
    
    # prompt for mbpp codellama (one shot steps)    
    # prompt = " Remember to respond with only the function definition, nothing else. You shall be provided steps to generate more accurate code. You are provided an example to understand the structure. \nTask: {description} \nTests:\n {tests}"
    
    #prompt for mbpp codellama (zero shot pseudo code)
    # prompt = " Remember to respond with only the function definition, nothing else. You shall be provided pseudocode to generate more accurate code. \nTask: {description} \nTests:\n {tests}"
    
    # prompt for mbpp codellama (one shot pseudo code)
    # prompt = " Remember to respond with only the function definition, nothing else. You shall be provided pseudocode to generate more accurate code. You are provided an example to understand the structure. \nTask: {description} \nTests:\n {tests}"
    
    # prompt = : "You are an expert Python programmer, and you need to respond with code to complete the following task. Remember to respond with only the function definition, nothing else. You shall be provided pseudocode to generate more accurate code. You are provided an example to understand the structure. \nTask: {description} \nTests:\n {tests}"

    instruction = prompt.format(description=description, tests="\n".join(test_cases))
    return instruction

def load_data_mbpp():
    dataset_name = 'mbpp'
    data_path_mapping = {
            "mbpp": "./data/mbpp.jsonl",
            "humaneval": "./data/HumanEval.jsonl.gz"
            }
    data_path = data_path_mapping[dataset_name]
    # data_path = "/home/mila/m/megh.thakkar/CodeCapybara/main/data/mbpp.jsonl"
    data = []
    if data_path.endswith(".jsonl.gz"):
        with gzip.open(data_path, "rt") as f:
            data = [json.loads(line) for line in f]
    elif data_path.endswith(".json"):
        with open(data_path, "r") as f:
            data = json.load(f)
    else:
        with open(data_path, "r") as f:
            data = [json.loads(line) for line in f]

    instructions = []
    task_ids = []
    if dataset_name == "mbpp":
        data = list(filter(lambda x: x["task_id"] in range(11, 511), data))
        instructions = list(map(create_mbpp_instruction, data))
        task_ids = list(map(lambda x: x["task_id"], data))
    else:
        task_ids = [ex["task_id"] for ex in data]
        instructions = [ex["prompt"] for ex in data]

    return task_ids, instructions

# reference: https://github.com/declare-lab/instruct-eval/blob/main/human_eval/main.py#L35
def filter_code(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    completion = completion.lstrip("\n")
    return completion.split("\n\n")[0]


def fix_indents(text: str) -> str:
    return text.replace("\t", "    ")


def split_batch(samples: list[str], size=4):
    mini_batches = []

    for i in range(0, len(samples), size):
        mini_batches.append(samples[i : i + size])

    return mini_batches


def run_eval(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_samples_per_task: int,
    out_path: str,
    generate_batch_completion: BatchGenerator,
    length: int,
    format_tabs: bool = False,
):
    problems = read_problems()
    problems = dict(itertools.islice(problems.items(), length))
    samples = []
    pbar = tqdm(total=len(problems) * num_samples_per_task)
    counter = 0
    for task_id in problems:
        if format_tabs:
            prompt = problems[task_id]["prompt"].replace("    ", "\t")
        else:
            prompt = problems[task_id]["prompt"]

        batch_completions = generate_batch_completion(
            model, tokenizer, prompt, num_samples_per_task, counter
        )

        for sample in batch_completions:
            result = dict(
                task_id=task_id,
                completion=sample,
            )

            samples += [result]

        pbar.update(num_samples_per_task)
        counter += 1

    write_jsonl(out_path, samples)

def run_eval_mbpp(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    num_samples_per_task: int,
    out_path: str,
    generate_batch_completion: BatchGenerator,
    length: int,
    format_tabs: bool = False,
):
    prompter = Prompter("")
    task_ids, instructions = load_data_mbpp()
    # problems = [prompter.generate_prompt(instruction) for instruction in instructions] # uncoment if not using gpt
    problems = instructions # comment if not using gpt  
    problems = problems[:length]    
    samples = []
    pbar = tqdm(total=len(problems) * num_samples_per_task)

    counter=0
    for i, prompt in enumerate(problems):
        # if format_tabs:
        #     prompt = problems[task_ids[i]]["prompt"].replace("    ", "\t")
        # else:
        #     prompt = problems[task_ids[i]]["prompt"]

        batch_completions = generate_batch_completion(
            model, tokenizer, prompt, num_samples_per_task, counter
        )
        # print (batch_completions)
        for sample in batch_completions:
            result = dict(
                task_id=task_ids[i],
                # trg_prediction=prompter.get_response(sample),
                trg_prediction=sample,
            )

            samples += [result]

        pbar.update(num_samples_per_task)
        counter += 1

    write_jsonl(out_path, samples)

if __name__ == "__main__":
    _, instructions = load_data_mbpp()
    print(instructions[:5])