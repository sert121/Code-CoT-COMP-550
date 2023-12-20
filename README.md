## Code Repository for Code-COT: Does Adapting Chain-of-Thought Help with Code Generation? |  COMP-550 Project (Fall 2023)

1. Installation:
```bash
$ pip install -r requirements.txt
```


2. HumanEval:
    
    2.1: Generate the test results by the model you'd like between codellama 7b and 34b Instruct variants:
    ```bash
    $ python eval_codellama_humaneval.py --model_name codellama/CodeLlama-7b-Instruct-hf --length 100
    ```
    2.2 To evaluate for a number greater/lesser than 100, you would need to change the length on line 57 in human-eval/human_eval/evaluation.py to match the length set.
    ```bash
    $ evaluate_functional_correctness results/codellama/humaneval_CodeLlama-7b-Python-hf_100.jsonl
    ```

    2.3 Change prompt: Different prompts are provided in the eval_codellama_human_eval.py (more details below).
3. MBPP:

    3.1 Generate:
    ```bash
    $ python eval_codellama_mbpp.py --model_name codellama/CodeLlama-7b-Instruct-hf --length 100
    ```
    2.2 Evaluate: # you may need to change the output directory of the model as per your choice of model in ```eval_mbpp.py``` line 254
    ```bash
    $ python eval_mbpp.py
    ```
