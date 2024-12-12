# git+https://github.com/HydroXai/GPTFuzz.git <- add to requirements

import time
import json
import os
import argparse
import random
from collections import namedtuple

import fire
import pandas as pd

# Local application/library specific imports
from gptfuzzer.llm import OpenAILLM, LocalVLLM, LocalLLM
from gptfuzzer.utils.predict import RoBERTaPredictor
from gptfuzzer.fuzzer.selection import MCTSExploreSelectPolicy
from gptfuzzer.fuzzer.mutator import (
    MutateRandomSinglePolicy, OpenAIMutatorCrossOver, OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar, OpenAIMutatorRephrase, OpenAIMutatorShorten)
from gptfuzzer.fuzzer import GPTFuzzer


import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')
# Suppress warnings from a specific library
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')


Result = namedtuple('Result', 'prompt targetModel error', defaults=(None, None, None))


def load_model(target_model_path: str, judge_model_path: str, openai_model_path: str = 'gpt-4o-mini-2024-07-18'):
    openai_model = OpenAILLM(openai_model_path, os.getenv('OPENAI_API_KEY'))             
    target_vllm_model = LocalVLLM(target_model_path, gpu_memory_utilization=0.9)
    roberta_model = RoBERTaPredictor(judge_model_path, device='cuda')
    return openai_model, target_vllm_model, roberta_model


def run_gptfuzzer(
    seed_path: str,
    target_model: str,
    judge_model: str,
    goal_prompt: str,
    max_query: int = 500,
    max_jailbreak: int = 1,
    energy: int = 1,
):

    initial_seed = pd.read_csv(seed_path)['text'].tolist()
    openai_model, target_vllm_model, roberta_model = load_model(target_model, judge_model)

    fuzzer = GPTFuzzer(
        questions=[goal_prompt],
        target=target_vllm_model,
        predictor=roberta_model,
        initial_seed=initial_seed,
        mutate_policy=MutateRandomSinglePolicy([
            OpenAIMutatorCrossOver(openai_model, temperature=0.0),
            OpenAIMutatorExpand(openai_model, temperature=1.0),
            OpenAIMutatorGenerateSimilar(openai_model, temperature=0.5),
            OpenAIMutatorRephrase(openai_model),
            OpenAIMutatorShorten(openai_model)],
            concatentate=True,
        ),
        select_policy=MCTSExploreSelectPolicy(),
        energy=energy,
        max_jailbreak=max_jailbreak,
        max_query=max_query,
        generate_in_batch=True,
    )

    adv_prompt, adv_response = fuzzer.run()
    return adv_prompt


def run(seed_path: str, target_model: str, judge_model: str, goal_prompt: str, result_path: str, max_query: int = 20):
    try:
        adv_prompt = run_gptfuzzer(seed_path, target_model, judge_model, goal_prompt, max_query)
        target_model_name = os.path.basename(target_model)
        result = [Result(prompt=adv_prompt,
                        targetModel=target_model_name
                        )._asdict()]
    except Exception as e:
        result = [Result(prompt=goal_prompt,
                        error=f"An error was detected during the GPTFuzzer attack: {e}")._asdict()]
    with open(result_path, 'w', encoding="utf8") as f:
        json.dump(result, f)


if __name__ == '__main__':
    fire.Fire(run)
    