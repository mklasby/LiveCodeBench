import openai
from lcb_runner.runner.base_runner import BaseRunner


import time

import openai
from openai.types.chat import ChatCompletion


def make_request(
    client: openai.Client,
    prompt: str,
    model: str,
    **kwargs
) -> ChatCompletion:

    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        **kwargs
    )


def make_auto_request(*args, **kwargs) -> ChatCompletion:
    ret = None
    while ret is None:
        try:
            ret = make_request(*args, **kwargs)
        except openai.RateLimitError:
            print("Rate limit exceeded. Waiting...")
            time.sleep(5)
        except openai.APIConnectionError:
            print("API connection error. Waiting...")
            time.sleep(5)
        except openai.APIError as e:
            print(e)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            time.sleep(1)
    return ret


class VLLMServerRunner(BaseRunner):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.client = openai.OpenAI(base_url=args.base_url, api_key="EMPTY")
        self.client_kwargs = {
            "model": model.model_name,
            "max_tokens": self.args.max_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "n": self.args.n,
            "stop": self.args.stop,
        }

    def _run_single(self, prompt: str) -> list[str]:
        request = make_auto_request(
            self.client,
            prompt,
            model=self.model.model_name,
            **self.client_kwargs
        )
        
        response = self.client.completions.create(prompt=prompt, **self.client_kwargs)
        return [choice.text for choice in response.choices]

    def run_batch(self, prompts: list[str]) -> list[list[str]]:
        outputs = []
        remaining_prompts = []
        remaining_indices = []
        for prompt_index, prompt in enumerate(prompts):
            if self.args.use_cache and prompt in self.cache:
                if len(self.cache[prompt]) == self.args.n:
                    outputs[prompt_index] = self.cache[prompt]
                    continue
            remaining_prompts.append(prompt)
            remaining_indices.append(prompt_index)
        if remaining_prompts:
            vllm_outputs = []
            for prompt in remaining_prompts:
                vllm_outputs.append(self._run_single(prompt))
            if self.args.use_cache:
                assert len(remaining_prompts) == len(vllm_outputs)
                for index, remaining_prompt, vllm_output in zip(
                    remaining_indices, remaining_prompts, vllm_outputs
                ):
                    self.cache[remaining_prompt] = [o.text for o in vllm_output.outputs]
                    outputs[index] = [o.text for o in vllm_output.outputs]
            else:
                for index, vllm_output in zip(remaining_indices, vllm_outputs):
                    outputs[index] = [o.text for o in vllm_output.outputs]
        return outputs


