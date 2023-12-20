import os
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from core import filter_code, run_eval_mbpp, fix_indents
import os
import torch
import json

# TODO: move to python-dotenv
# add hugging face access token here
TOKEN = "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"


def load_json(file_path="./mbpp_examples_magicoder_reform_v1.json"):
    with open(file_path, "r") as json_file:
        data_dict = json.load(json_file)

    list_prompts = []
    for k, v in data_dict.items():
        list_prompts.append(v)
    # sorted_dict = {int(key): value for key, value in sorted(data_dict.items())}
    # sorted_value_list = [value for key, value in sorted_dict.items()]
    return list_prompts


# ZERO SHOT + STEPS
def construct_codellama_prompt(problem, thought):
    problem = problem.split('[/INST]')[0]
    PROMPT = f'{problem} \nSteps:\n {thought} \nCode:[/INST]'
    return PROMPT


# ZERO SHOT + PSEUDOCODE
def construct_codellama_pseudo_prompt(problem, thought):
    problem = problem.split('[/INST]')[0]
    PROMPT = f'{problem} \nPseudocode:\n {thought} \nCode:\n[/INST]'
    return PROMPT



# ONE SHOT + PSSEUDOCODE
def construct_codellama_pseudo_prompt_example(problem, thought):
    problem = problem.split('[/INST]')[0]
    example = '''
    EXAMPLE STARTS HERE
        Task: Write a program that extracts all substrings of length n from a given string.
        Tests:
        assert find_substrings("abc", 2) == ["ab", "bc"]
        assert find_substrings("abc", 3) == ["abc"]
        assert find_substrings("abc", 4) == []
        
        Pseudocode:
        function extract_substrings(string, n)
            # Initialize an empty list for substrings
            Initialize substrings as an empty list

            # Loop from start to the point where substring of length n can be extracted
            for every index in the string
                # Add substring of length n to the list
                append string[i:i + n] to substrings

            # Return the list of substrings
            return the substrings
        Code:
        def find_substrings(string, n):
            substrings = []
            for i in range(len(string) - n + 1):
                substrings.append(string[i:i + n])
            return substrings
    EXAMPLE ENDS HERE
    '''

    # used for benchmarking
    PROMPT = example + f'{problem} \nPseudocode:\n{thought} \nCode:\n[/INST]'

    # v2 used for experimentation
    PROMPT = f'{problem}\n {example} \nPseudocode:\n{thought} \nCode:\n[/INST]'
    return PROMPT



# ONE SHOT  + STEPS PROMPT
def construct_codellama_prompt_steps(problem, thought):
    problem = problem.split('[/INST]')[0]
    example = '''
EXAMPLE STARTS HERE
    Task: Write a program that extracts all substrings of length n from a given string.
    Tests:
    assert find_substrings("abc", 2) == ["ab", "bc"]
    assert find_substrings("abc", 3) == ["abc"]
    assert find_substrings("abc", 4) == []
    
    Steps:
    1. Initialize an empty list for substrings
    2. Loop from start to the point where substring of length n can be extracted
        a. Add substring of length n to the list
    3. Return the final list of substrings that was created
    Code:
    def find_substrings(string, n):
        substrings = []
        for i in range(len(string) - n + 1):
            substrings.append(string[i:i + n])
        return substrings
EXAMPLE ENDS HERE     
'''
    PROMPT = example + \
        f'{problem} \nSteps:\n {thought} \nCode:\n[/INST]'  # v1 that is used in baselines

    # v2
    PROMPT = f'{problem}\n {example} \nSteps:\n {thought} \nCode:\n[/INST]'

    return PROMPT


@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt, batch_size, counter
) -> list[str]:
    # to account for mbpp -delay , mbpp starts testing on samples from 10th sample onwards
    counter = counter + 10

    input_batch = [prompt for _ in range(batch_size)]
    # print ("Input: ", input_batch)
    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    # thoughts = load_json("./mbpp_examples_magicoder_reform_v1.json") # !NOTE for step by step examples

    
    thoughts = load_json("./mbpp_actualpsuedocode_magicoder_reform_v1.json") # !NOTE # for pseudocode examples

    example_thought = thoughts[counter]

    # input_batch = [example_thought for _ in range(batch_size)]
    list_thoughts = [thoughts[counter] for _ in range(batch_size)]

    # input_batch = [construct_codellama_pseudo_prompt(x, y) for x, y in zip(input_batch, list_thoughts)] #note for pseuducode

    # input_batch = [construct_codellama_prompt(x, y) for x, y in zip(input_batch, list_thoughts)] #note zero shot steps

    # input_batch = [construct_codellama_prompt_steps(x, y) for x, y in zip(input_batch, list_thoughts)] # note for step by step + one shot

    input_batch = [construct_codellama_pseudo_prompt_example(x, y) for x, y in zip(
        input_batch, list_thoughts)]  # note for pseudocode + one shot

    print("Sample prompt:\n", input_batch[0])
    # save to a file
    if counter == 100:
        with open("sample_prompt.txt", "w") as f:
            f.write("We are at 100th sample\n")
            f.write(input_batch[0])

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # model has no pad token
    )

    batch_completions = tokenizer.batch_decode(
        [ids for ids in generated_ids],
        skip_special_tokens=True,
        ignore_tokenization_space=True
    )

    return [completion for completion in batch_completions]


if __name__ == "__main__":
    # adjust for n = 10 etc
    parser = argparse.ArgumentParser(description="Eval model")
    # parser.add_argument('--model_name', type=str, default='codellama/CodeLlama-34b-Instruct-hf') # uncomment this line for 34b model

    parser.add_argument('--model_name', type=str,
                        default='codellama/CodeLlama-7b-Instruct-hf')    # uncomment this line for 7b model
    parser.add_argument('--length', type=int, default=100)
    args = parser.parse_args()
    print(args)

    num_samples_per_task = 5

    # output path
    out_path = "results/" + args.model_name.split('/')[0] + "/mbpp_" + args.model_name.split('/')[
        1] + '_' + str(args.length) + ".jsonl"
    print("Out path: ", out_path)
    os.makedirs("results/" + args.model_name.split('/')[0], exist_ok=True)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
    )

    print("Loading model...")
    model = torch.compile(
        AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
        )
        .eval()
        .to("cuda")
    )

    run_eval_mbpp(model, tokenizer, num_samples_per_task, out_path,
                  generate_batch_completion, args.length, True)
