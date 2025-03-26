# git+https://github.com/HydroXai/GPTFuzz.git <- add to requirements

import fire
import json
import pandas as pd
import traceback
import warnings

from collections import namedtuple
from gptfuzzer.fuzzer import GPTFuzzer
from gptfuzzer.fuzzer.mutator import (
    MutateRandomSinglePolicy,
    OpenAIMutatorCrossOver,
    OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar,
    OpenAIMutatorRephrase,
    OpenAIMutatorShorten
)
from gptfuzzer.fuzzer.selection import MCTSExploreSelectPolicy
from gptfuzzer.llm import LLM, LocalLLM
from gptfuzzer.utils.predict import RoBERTaPredictor


warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')


MAX_RETRY_COUNT = 2
MAX_QUERY = 50


Result = namedtuple('Result', 'response error', defaults=(None, None))


def run_gptfuzzer(
    seed_path: str,
    openai_model: LLM,
    target_model: LLM,
    judge_model: RoBERTaPredictor,
    goal_prompt: str,
    max_query: int = 500,
    max_jailbreak: int = 1,
    energy: int = 1,
):

    initial_seed = pd.read_csv(seed_path)['text'].tolist()

    fuzzer = GPTFuzzer(
        questions=[goal_prompt],
        target=target_model,
        predictor=judge_model,
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


def main(input_path: str, output_path: str) -> None:
    results = []
    with open(input_path, mode='r') as f:
        samples = json.load(f)
        seed_path = samples.get("seedPath")
        open_ai_replacement_model_path = samples.get("openAiReplacementModelPath")
        target_model_path = samples.get("targetModelPath")
        judge_model_path = samples.get("judgeModelPath")

        open_ai_model = LocalLLM(open_ai_replacement_model_path)
        target_model = LocalLLM(target_model_path)
        judge_model = RoBERTaPredictor(judge_model_path)

        for goal_prompt in samples.get("goalPrompts"):
            count = 0
            while count < MAX_RETRY_COUNT:
                try:
                    adv_prompt = run_gptfuzzer(seed_path, open_ai_model, target_model, judge_model, goal_prompt, MAX_QUERY)
                    results.append(Result(response=adv_prompt)._asdict())
                    break
                except Exception as e:
                    print(traceback.format_exc())

                    if count == MAX_RETRY_COUNT - 1:
                        results = [Result(error=f"An error was detected during the AutoDAN attack: {e}")._asdict()]
                    count += 1
    with open(output_path, 'w', encoding="utf8") as f:
        json.dump(results, f)


if __name__ == '__main__':
    fire.Fire(main)
