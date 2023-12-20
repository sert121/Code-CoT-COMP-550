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
from core import filter_code, run_eval, fix_indents
import os
import torch
import json
import re
import textwrap
'''
A fresh fork of eval_model_humaneval.
'''

# add hugging face access token here
TOKEN = ""  # insert huggingface token here


def extract_last_comment_from_code(code):
    # Use regular expressions to find comments within triple-quoted strings
    pattern = r'(\'\'\'|\"\"\")(.+?)\1'
    comments = re.findall(pattern, code, re.DOTALL)

    if comments:
        # Extract the first comment from the list of comments
        first_comment = comments[-1][1].strip()
        return first_comment

    return None


def load_json(file_path=""):
    with open(file_path, "r") as json_file:
        data_dict = json.load(json_file)

    list_prompts = []
    for k, v in data_dict.items():
        list_prompts.append(v)
    return list_prompts


# SIMPLE VARIANT OF THE PROMPT
def construct_codellama_prompt(problem, thought):
    PROMPT_DEF = f'''[INST]
You are a software engineer.Your task is to write a Python function to solve a task given below.
{problem}

    ### Some thinking process to help:
    {thought}
[/INST]
'''
    return PROMPT_DEF


# SIMPLE VARIANT OF THE PROMPT #2
def construct_codellama_prompt_v2(problem, thought):
    PROMPT_DEF = f'''[INST]
You are a software engineer.Your task is to write a Python function to solve a task given below.
{problem}

Psuedocode to help:
{thought}
[/INST]
'''
    return PROMPT_DEF


# PROMPT  FOR ONESHOT STEPS + EXAMPLE (VARIANT 1)
def construct_codellama_prompt_oneshot_examples(problem, thought):
    PROMPT_DEF = f'''[INST]
Below is an instruction that describes a task. Write a response that appropriately completes the request:
Remember to respond with only the function definition, nothing else. You shall be provided steps to generate more accurate code. 
You are provided an example to understand the structure.
'''
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

    PROMPT = PROMPT_DEF + example + \
        f' Task:\n {problem} \nSteps:\n {thought} \nCode:\n[/INST]'
    return PROMPT


# PROMPT  FOR ONESHOT PSEUDOCODE + EXAMPLE
def construct_codellama_comment_prompt_one_shot_psuedocode(problem, thought):
    # main_comment = extract_first_comment_from_code(problem)
    function_code = problem
    main_comment_direct = extract_last_comment_from_code(problem)

    pattern_corrected = r"(\'\'\'|\"\"\")[\s\S]+?(\1)"
    additional_comment = "        Here's a solving process to help:\n" + thought

    example = '''
EXAMPLE STARTS HERE
    import typing
    def find_substrings(string, n):
    """
     Write a program that extracts all substrings of length n from a given string.
     Here's a pseudocode to help:
      function find_substrings (nput_string, input_n)
        initialize empty list substrings

        for i in range from 0 to length(input_string) - input_n
            append substring from input_string[i] to input_string[i + input_n - 1] to substrings

        return substrings
    """
    def find_substrings(string, n):
        substrings = []
        for i in range(len(string) - n + 1):
            substrings.append(string[i:i + n])
        return substrings
EXAMPLE ENDS HERE     
'''

    additional_comment = "        Here's a solving process to help:\n" + thought + \
        "You're also provided an example above to understand the structure better" + \
        example  # few shot steps + oneshot example

    additional_comment_lines = additional_comment.splitlines()

    # Add indentation to each line (4 spaces)
    indented_additional_comment = "\n        ".join(additional_comment_lines)
    # Now append this indented comment to the main comment
    combined_comment_with_indentation = f"{main_comment_direct.strip()}\n\n    {indented_additional_comment}"

    # Replace the first multiline comment with the indented combined comment
    # Using the corrected regular expression pattern
    modified_function_code_with_indentation = re.sub(
        pattern_corrected, f"\"\"\"{combined_comment_with_indentation}\"\"\"", function_code, count=1)

    l = modified_function_code_with_indentation
    # print(l), print("\n--\n"d)
    return l


# PROMPT  FOR ONESHOT STEPS + EXAMPLE (VARIANT 2)
def construct_codellama_comment_prompt_one_shot(problem, thought):
    # main_comment = extract_first_comment_from_code(problem)
    function_code = problem
    main_comment_direct = extract_last_comment_from_code(problem)

    pattern_corrected = r"(\'\'\'|\"\"\")[\s\S]+?(\1)"
    additional_comment = "        Here's a solving process to help:\n" + \
        thought  # few shot steps

    example = '''
EXAMPLE STARTS HERE
    import typing
    def find_substrings(string, n):
    """
     Write a program that extracts all substrings of length n from a given string.
     Here's a solving process to help:
        1. Initialize an empty list for substrings
        2. Loop from start to the point where substring of length n can be extracted
            a. Add substring of length n to the list
        3. Return the final list of substrings that was created

    """
    def find_substrings(string, n):
        substrings = []
        for i in range(len(string) - n + 1):
            substrings.append(string[i:i + n])
        return substrings
EXAMPLE ENDS HERE     
'''

    additional_comment = "        Here's a solving process to help:\n" + thought + \
        "You're also provided the following example to understand the structure better" + \
        example  # few shot steps + oneshot example

    additional_comment_lines = additional_comment.splitlines()

    # Add indentation to each line (4 spaces)
    indented_additional_comment = "\n        ".join(additional_comment_lines)
    # Now append this indented comment to the main comment
    combined_comment_with_indentation = f"{main_comment_direct.strip()}\n\n    {indented_additional_comment}"

    # Replace the first multiline comment with the indented combined comment
    # Using the corrected regular expression pattern
    modified_function_code_with_indentation = re.sub(
        pattern_corrected, f"\"\"\"{combined_comment_with_indentation}\"\"\"", function_code, count=1)

    l = modified_function_code_with_indentation
    # print(l), print("\n--\n"d)
    return l


# ZERO SHOT STEPS/PSEUDOCODE (depends on how you are loading the thoughts)
def construct_codellama_comment_prompt(problem, thought):
    # main_comment = extract_first_comment_from_code(problem)
    function_code = problem
    main_comment_direct = extract_last_comment_from_code(problem)

    pattern_corrected = r"(\'\'\'|\"\"\")[\s\S]+?(\1)"
    # additional_comment = "        Here's a solving process to help:\n" + thought # few shot steps
    additional_comment = "        Here's a Psuedocode to help:\n" + thought  # few shot code

    # additional_comment = "        Here's a solving process to help:\n" + thought + "You're also provided the following example to understand the structure better" + example  # few shot steps + oneshot example

    additional_comment_lines = additional_comment.splitlines()

    # Add indentation to each line (4 spaces)
    indented_additional_comment = "\n        ".join(additional_comment_lines)
    # Now append this indented comment to the main comment
    combined_comment_with_indentation = f"{main_comment_direct.strip()}\n\n    {indented_additional_comment}"

    # Replace the first multiline comment with the indented combined comment
    # Using the corrected regular expression pattern
    modified_function_code_with_indentation = re.sub(
        pattern_corrected, f"\"\"\"{combined_comment_with_indentation}\"\"\"", function_code, count=1)

    l = modified_function_code_with_indentation
    return l


@torch.inference_mode()
def generate_batch_completion(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt, batch_size, counter
) -> list[str]:
    input_batch = [prompt for _ in range(batch_size)]

    # thoughts = load_json(
    #     "./humaneval_actualpsuedocode_magicoder_reform_v1.json") # !NOTE: for psuedocode examples
   
    thoughts = load_json("./humaneval_steps_magicoder.json")  # !NOTE: for steps

    list_thoughts = [thoughts[counter] for _ in range(batch_size)]

    # input_batch = [construct_codellama_prompt(x, y) for x, y in zip(input_batch, list_thoughts)] # simple prompt with steps

    # input_batch = [construct_codellama_comment_prompt(x, y) for x, y in zip(input_batch, list_thoughts)] #  prompt with steps/psuedocode zero shot
    
    input_batch = [construct_codellama_comment_prompt(x, y) for x, y in zip(
        input_batch, list_thoughts)]  # ZERO SHOT STEPS/PSEUDOCODE
    
    # input_batch = [construct_codellama_prompt_oneshot_examples(x, y) for x, y in zip(input_batch, list_thoughts)] # prompt with steps + oneshot example
    
    # input_batch = [construct_codellama_comment_prompt_one_shot_psuedocode(
    #     x, y) for x, y in zip(input_batch, thoughts)] # prompt with pseudocode + oneshot example

    # input_batch = [construct_codellama_prompt_oneshot_examples(x, y) for x, y in zip(input_batch, list_thoughts)]

    for i in range(len(input_batch)):
        if "[PYTHON]" in input_batch[i]:
            pattern = r"\[PYTHON\](.*?)\[/PYTHON\]"
            match = re.search(pattern, input_batch[i], re.DOTALL)
            extracted_text = match.group(
                1).strip() if match else "No match found"
            if extracted_text != "No match found":
                input_batch[i] = extracted_text

    # print sample input to the model
    # print("Sample Input:\n", input_batch[0])

    inputs = tokenizer(input_batch, return_tensors="pt").to(model.device)
    input_ids_cutoff = inputs.input_ids.size(dim=1)

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=1024,
        temperature=0.2,
        top_p=0.95,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,  # model has no pad token
    )

    batch_completions = tokenizer.batch_decode(
        [ids[input_ids_cutoff:] for ids in generated_ids],
        skip_special_tokens=True,
    )
    batch_completions = [fix_indents(completion)
                         for completion in batch_completions]
    # batch_completions = [textwrap.dedent(completion) for completion in batch_completions] # fixing some indentation issues
    return batch_completions


if __name__ == "__main__":
    # adjust for n = 10 etc

    parser = argparse.ArgumentParser(description="Eval model")
    # parser.add_argument('--model_name', type=str, default='codellama/CodeLlama-7b-Instruct-hf') # uncomment this line for 7b model
    parser.add_argument('--model_name', type=str,
                        default='codellama/CodeLlama-34b-Instruct-hf') # uncomment this line for 34b model
    parser.add_argument('--length', type=int, default=10)
    args = parser.parse_args()
    print(args)

    num_samples_per_task = 5

    #output path
    out_path = "results/" + args.model_name.split('/')[0] + "/humaneval_" + args.model_name.split('/')[
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

    run_eval(
        model,
        tokenizer,
        num_samples_per_task,
        out_path,
        generate_batch_completion,
        args.length,
        True,
    )
