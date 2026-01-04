from typing import List, Dict
import tiktoken


def _get_logit_bias(model_name: str, allowed_tokens: List[str]) -> Dict[int, int]:
    enc = tiktoken.encoding_for_model(model_name)
    allowed_token_ids = []
    for token in allowed_tokens:
        encoded = enc.encode(token)
        if len(encoded) == 1:
            allowed_token_ids.append(encoded[0])
        else:
            # tokens that encode to multiple IDs
            print(f'Skipping token "{token}" (encodes to multiple IDs: {encoded})')
    return {tid: 100 for tid in allowed_token_ids}
