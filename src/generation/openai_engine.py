"""Generation and clause-check scoring with API models through the OpenAI Responses API."""
from __future__ import annotations

import math

from openai import OpenAI

TOP_LOGPROBS = 20


class OpenAIEngine:
    """Generation and clause-check scoring with an API model."""
    def __init__(self, model: str = "gpt-6-luna"):
        """Create the API client for the given model."""
        self.client = OpenAI()
        self.model = model

    def _text(self, messages, *, temperature: float, top_p: float, max_tokens: int) -> str:
        """Output text of one request without reasoning."""
        response = self.client.responses.create(model=self.model, input=messages, reasoning={"effort": "none"},
                                                temperature=temperature, top_p=top_p, max_output_tokens=max_tokens,
                                                store=False)
        return response.output_text

    def generate(self, message_batches, *, num_samples: int, temperature: float, top_p: float,
                 max_tokens: int, seeds=None) -> list[tuple[str, list[str]]]:
        """(greedy response, sampled responses) per question."""
        out = []
        for messages in message_batches:
            greedy = self._text(messages, temperature=0.0, top_p=1.0, max_tokens=max_tokens)
            samples = [self._text(messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
                       for _ in range(num_samples)]
            out.append((greedy, samples))
        return out

    def p_yes(self, message_batches) -> list[float | None]:
        """P(Yes) over the Yes and No variants among the top log-probabilities of the first output token."""
        values = []
        for messages in message_batches:
            response = self.client.responses.create(
                model=self.model, input=messages, reasoning={"effort": "none"}, temperature=0.0, top_p=1.0,
                max_output_tokens=16, top_logprobs=TOP_LOGPROBS, include=["message.output_text.logprobs"],
                store=False)
            values.append(_p_yes(response))
        return values


def _p_yes(response) -> float | None:
    """P(Yes) from the first output token of a Yes/No response, or None when unavailable."""
    for item in response.output:
        if getattr(item, "type", None) != "message":
            continue
        for content in item.content:
            for entry in getattr(content, "logprobs", None) or []:
                if not str(entry.token).strip():
                    continue
                candidates = {str(entry.token): float(entry.logprob)}
                for alternative in entry.top_logprobs or []:
                    candidates.setdefault(str(alternative.token), float(alternative.logprob))
                label = lambda token: token.strip().casefold()
                if label(str(entry.token)) not in {"yes", "no"}:
                    return None
                yes = [lp for tok, lp in candidates.items() if label(tok) == "yes"]
                no = [lp for tok, lp in candidates.items() if label(tok) == "no"]
                if not yes or not no:
                    return None
                top = max(yes + no)
                yes_mass = sum(math.exp(lp - top) for lp in yes)
                no_mass = sum(math.exp(lp - top) for lp in no)
                return yes_mass / (yes_mass + no_mass)
    return None
