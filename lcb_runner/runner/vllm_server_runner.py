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
        messages=[{"role": "user", "content": prompt}],
        **kwargs
    )


def make_auto_request(*args, **kwargs) -> ChatCompletion:
    completion = None
    while completion is None:
        try:
            completion = make_request(*args, **kwargs)
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
    return completion


class VLLMServerRunner(BaseRunner):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.model_name = model.model_name if args.local_model_path is None else args.local_model_path
        self.enable_thinking = args.enable_thinking
        self.client = openai.OpenAI(base_url=args.base_url, api_key="EMPTY")
        self.client_kwargs = {
            "model": self.model_name,
            "max_tokens": self.args.max_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "n": self.args.n,
        }
        if not self.enable_thinking:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
            self.client_kwargs["extra_body"] = extra_body

    def _run_single(self, prompt: str) -> list[str]:
        completion = make_auto_request(
            self.client,
            prompt,
            **self.client_kwargs
        )
        return [c.message.content for c in completion.choices]

    # def run_batch_v2(self, prompts: list[str | list[dict[str, str]]]) -> list[list[str]]:
    #         outputs = []
    #         arguments = [
    #             (
    #                 prompt,
    #                 self.cache,  ## pass the cache as argument for cache check
    #                 self.args,  ## pass the args as argument for cache check
    #                 self._run_single,  ## pass the _run_single method as argument because of multiprocessing
    #             )
    #             for prompt in prompts
    #         ]
    #         if self.args.multiprocess > 1:
    #             parallel_outputs = run_tasks_in_parallel(
    #                 self.run_single,
    #                 arguments,
    #                 self.args.multiprocess,
    #                 use_progress_bar=True,
    #             )
    #             for output in parallel_outputs:
    #                 if output.is_success():
    #                     outputs.append(output.result)
    #                 else:
    #                     print("Failed to run the model for some prompts")
    #                     print(output.status)
    #                     print(output.exception_tb)
    #                     outputs.extend([""] * self.args.n)
    #         else:
    #             outputs = [self.run_single(argument) for argument in tqdm(arguments)]

    #         if self.args.use_cache:
    #             for prompt, output in zip(prompts, outputs):
    #                 if isinstance(prompt, list):
    #                     prompt_cache = json.dumps(prompt)
    #                 elif isinstance(prompt, tuple):
    #                     prompt_cache = prompt[0] + json.dumps(prompt[1])
    #                 else:
    #                     prompt_cache = prompt
    #                 self.cache[prompt_cache] = output  ## save the output to cache

    #         return outputs


