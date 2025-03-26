import concurrent.futures
import google.generativeai as palm
import logging
import time
import torch

from accelerate import Accelerator
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from fastchat.model import load_model, get_conversation_template
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerFast
from vllm import LLM as vllm
from vllm import SamplingParams


accelerator = Accelerator()
quantization_config = BitsAndBytesConfig(
    load_in_8Bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16
)


class LLM:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def generate(self, prompt):
        raise NotImplementedError("LLM must implement generate method.")

    def predict(self, sequences):
        raise NotImplementedError("LLM must implement predict method.")


class LocalLLM(LLM):
    def __init__(self, model_path) -> None:
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config, device_map="auto")
        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.system_message = None

    def set_system_message(self, conv_temp):
        if self.system_message is not None:
            conv_temp.set_system_message(self.system_message)

    def generate(self, prompt, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        messages = []
        if self.system_message is not None:
            messages.append({"role": "system", "content": self.system_message})
        messages.append({"role": "user", "content": prompt})
        input_ids = self.tokenizer.apply_chat_template(messages,
                                                    add_generation_prompt=True,
                                                    return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(input_ids,
                                        max_new_tokens=512,
                                        do_sample=True,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        temperature=0.8)
        outputs = generated_ids[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return [response]


    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, max_tokens, n,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results

    # @torch.inference_mode()
    # def generate_batch(self, prompts, temperature=0.01, max_tokens=512, repetition_penalty=1.0, batch_size=16):
    #     prompt_inputs = []
    #     for prompt in prompts:
    #         conv_temp = get_conversation_template(self.model_path)
    #         self.set_system_message(conv_temp)

    #         conv_temp.append_message(conv_temp.roles[0], prompt)
    #         conv_temp.append_message(conv_temp.roles[1], None)

    #         prompt_input = conv_temp.get_prompt()
    #         prompt_inputs.append(prompt_input)

    #     if self.tokenizer.pad_token == None:
    #         self.tokenizer.pad_token = self.tokenizer.eos_token
    #     self.tokenizer.padding_side = "left"
    #     input_ids = self.tokenizer(prompt_inputs, padding=True).input_ids
    #     # load the input_ids batch by batch to avoid OOM
    #     outputs = []
    #     for i in range(0, len(input_ids), batch_size):
    #         output_ids = self.model.generate(
    #             torch.as_tensor(input_ids[i:i+batch_size]).cuda(),
    #             do_sample=False,
    #             temperature=temperature,
    #             repetition_penalty=repetition_penalty,
    #             max_new_tokens=max_tokens,
    #         )
    #         output_ids = output_ids[:, len(input_ids[0]):]
    #         outputs.extend(self.tokenizer.batch_decode(
    #             output_ids, skip_special_tokens=True, spaces_between_special_tokens=False))
    #     return outputs


class LocalVLLM(LLM):
    def __init__(self,
                 model_path,
                 gpu_memory_utilization=0.95,
                 system_message=None
                 ):
        super().__init__()
        self.model_path = model_path
        self.model = vllm(
            self.model_path, gpu_memory_utilization=gpu_memory_utilization)
        
        if system_message is None and 'Llama-2' in model_path:
            # monkey patch for latest FastChat to use llama2's official system message
            self.system_message = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
            "Please ensure that your responses are socially unbiased and positive in nature.\n\n" \
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
            "If you don't know the answer to a question, please don't share false information."
        else:
            self.system_message = system_message

    def set_system_message(self, conv_temp):
        if self.system_message is not None:
            conv_temp.set_system_message(self.system_message)

    def generate(self, prompt, temperature=0, max_tokens=512):
        prompts = [prompt]
        return self.generate_batch(prompts, temperature, max_tokens)

    def generate_batch(self, prompts, temperature=0, max_tokens=512):
        prompt_inputs = []
        for prompt in prompts:
            conv_temp = get_conversation_template(self.model_path)
            self.set_system_message(conv_temp)

            conv_temp.append_message(conv_temp.roles[0], prompt)
            conv_temp.append_message(conv_temp.roles[1], None)

            prompt_input = conv_temp.get_prompt()
            prompt_inputs.append(prompt_input)

        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        results = self.model.generate(
            prompt_inputs, sampling_params, use_tqdm=False)
        outputs = []
        for result in results:
            outputs.append(result.outputs[0].text)
        return outputs


class BardLLM(LLM):
    def generate(self, prompt):
        return

class PaLM2LLM(LLM):
    def __init__(self,
                 model_path='chat-bison-001',
                 api_key=None,
                 system_message=None
                ):
        super().__init__()
        
        if len(api_key) != 39:
            raise ValueError('invalid PaLM2 API key')
        
        palm.configure(api_key=api_key)
        available_models = [m for m in palm.list_models()]
        for model in available_models:
            if model.name == model_path:
                self.model_path = model
                break
        self.system_message = system_message
        # The PaLM-2 has a great rescriction on the number of input tokens, so I will release the short jailbreak prompts later
        
    def generate(self, prompt, temperature=0, n=1, max_trials=1, failure_sleep_time=1):
        for _ in range(max_trials):
            try:
                results = palm.chat(
                    model=self.model_path,
                    prompt=prompt,
                    temperature=temperature,
                    candidate_count=n,
                )
                return [results.candidates[i]['content'] for i in range(n)]
            except Exception as e:
                logging.warning(
                    f"PaLM2 API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]
    
    def generate_batch(self, prompts, temperature=0, n=1, max_trials=1, failure_sleep_time=1):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, n,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results

class ClaudeLLM(LLM):
    def __init__(self,
                 model_path='claude-instant-1.2',
                 api_key=None
                ):
        super().__init__()
        
        if len(api_key) != 108:
            raise ValueError('invalid Claude API key')
        
        self.model_path = model_path
        self.api_key = api_key
        self.anthropic = Anthropic(
            api_key=self.api_key
        )

    def generate(self, prompt, max_tokens=512, max_trials=1, failure_sleep_time=1):
        
        for _ in range(max_trials):
            try:
                completion = self.anthropic.completions.create(
                    model=self.model_path,
                    max_tokens_to_sample=300,
                    prompt=f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}",
                )
                return [completion.completion]
            except Exception as e:
                logging.warning(
                    f"Claude API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" "]
    
    def generate_batch(self, prompts, max_tokens=512, max_trials=1, failure_sleep_time=1):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, max_tokens,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results

class OpenAILLM(LLM):
    def __init__(self,
                 model_path,
                 api_key=None,
                 system_message=None
                ):
        super().__init__()

        if not api_key.startswith('sk-'):
            raise ValueError('OpenAI API key should start with sk-')
        self.client = OpenAI(api_key = api_key)
        self.model_path = model_path
        self.system_message = system_message if system_message is not None else "You are a helpful assistant."

    def generate(self, prompt, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        for _ in range(max_trials):
            try:
                results = self.client.chat.completions.create(
                    model=self.model_path,
                    messages=[
                        {"role": "system", "content": self.system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                )
                return [results.choices[i].message.content for i in range(n)]
            except Exception as e:
                logging.warning(
                    f"OpenAI API call failed due to {e}. Retrying {_+1} / {max_trials} times...")
                time.sleep(failure_sleep_time)

        return [" " for _ in range(n)]

    def generate_batch(self, prompts, temperature=0, max_tokens=512, n=1, max_trials=10, failure_sleep_time=5):
        results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.generate, prompt, temperature, max_tokens, n,
                                       max_trials, failure_sleep_time): prompt for prompt in prompts}
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        return results
