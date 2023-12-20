import fire
import sys

from human_eval.data import HUMAN_EVAL
from human_eval.evaluation import evaluate_functional_correctness
import argparse,json

def entry_point(
    sample_file: str,
    k: str = "1,3,5",
    n_workers: int = 4,
    timeout: float = 3.0,
    problem_file: str = HUMAN_EVAL,
):

    file_path = "./human-eval/human_eval/conf.json"
    with open(file_path, "r") as json_file:
        d = json.load(json_file)
    
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(sample_file, k, n_workers, timeout, problem_file, d['length'])
    print(results)


def main():

    fire.Fire(entry_point)


sys.exit(main())
