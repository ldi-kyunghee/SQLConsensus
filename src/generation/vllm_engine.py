"""Generation and clause-check scoring with open-weight models served by vLLM."""
from __future__ import annotations

import math

from vllm import LLM, SamplingParams


def _token_ids(value) -> list[int]:
    """Flatten tokenizer output into one list of token ids."""
    while isinstance(value, dict) or hasattr(value, "keys"):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    value = list(value)
    while value and isinstance(value[0], (list, tuple)):
        value = list(value[0])
    return [int(v) for v in value]


def _single_token_id(tokenizer, text: str) -> int:
    """Token id of a word that the tokenizer encodes as one token."""
    for variant in (text, f" {text}"):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    raise ValueError(f"{text!r} is not a single token for this tokenizer")


class VLLMEngine:
    """Generation and clause-check scoring with an open-weight model served by vLLM."""
    def __init__(self, model: str, *, max_model_len: int = 16384, tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9, dtype: str = "bfloat16"):
        """Load the model and the token ids of Yes and No."""
        self.llm = LLM(model=model, dtype=dtype, max_model_len=max_model_len,
                       tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization)
        self.tokenizer = self.llm.get_tokenizer()
        self.yes_id = _single_token_id(self.tokenizer, "Yes")
        self.no_id = _single_token_id(self.tokenizer, "No")

    def generate(self, message_batches, *, num_samples: int, temperature: float, top_p: float,
                 max_tokens: int, seeds: list[int]) -> list[tuple[str, list[str]]]:
        """(greedy response, sampled responses) per question."""
        greedy = self.llm.chat(message_batches, sampling_params=SamplingParams(temperature=0.0, max_tokens=max_tokens),
                               use_tqdm=False)
        sampled = self.llm.chat(message_batches, use_tqdm=False, sampling_params=[
            SamplingParams(n=num_samples, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
            for seed in seeds])
        return [(g.outputs[0].text, [o.text for o in s.outputs]) for g, s in zip(greedy, sampled)]

    def p_yes(self, message_batches) -> list[float]:
        """P(Yes) normalized over the single-token responses Yes and No, from exact prompt log-probabilities."""
        prompts, tokens = [], []
        for messages in message_batches:
            prefix = _token_ids(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
            for token in (self.yes_id, self.no_id):
                prompts.append({"prompt_token_ids": [*prefix, token]})
                tokens.append(token)
        outputs = self.llm.generate(prompts, sampling_params=SamplingParams(temperature=0.0, max_tokens=1,
                                                                            prompt_logprobs=0), use_tqdm=False)
        logprobs = []
        for output, token in zip(outputs, tokens):
            item = output.prompt_logprobs[-1][token]
            logprobs.append(float(getattr(item, "logprob", item)))
        values = []
        for yes, no in zip(logprobs[0::2], logprobs[1::2]):
            top = max(yes, no)
            values.append(math.exp(yes - top) / (math.exp(yes - top) + math.exp(no - top)))
        return values
