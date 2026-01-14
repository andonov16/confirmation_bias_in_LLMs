from typing import Dict, Callable
from collections import defaultdict


def canonicalize_token(token: str) -> str:
    token = token.replace(" ", "").strip()
    return "".join(sorted(token))


def get_logprobs_dict(
    logprobs_list: list,
    canonicalizer: Callable[[str], str] = canonicalize_token,
) -> Dict[str, float]:
    top_logprobs = defaultdict(float)

    for lp in logprobs_list:
        canon = canonicalizer(lp.token)
        top_logprobs[canon] += lp.logprob

    return dict(top_logprobs)
