from typing import Dict
import numpy as np
import requests as r

from src.model_clients.base_model_client import BaseModelClient


class OllamaAIClient(BaseModelClient):
    def __init__(self,
                 api_key: str,
                 base_url: str,
                 model_name: str):
        self.model_name = model_name
        self.base_url = base_url

    def send_prompt_get_response(
        self,
        system_prompt: str,
        user_prompt: str,
        allowed_tokens: list[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, float]:
    

        url = "http://" + self.base_url + "/api/generate"

        args = {
            "model": self.model_name,
            "prompt": f"{system_prompt} \n\n {user_prompt}",
            "logprobs": True,
            "top_logprobs": len(allowed_tokens),
            "options": {
                "temperature": 0,
                "num_predict": 1
            }
        }
        try:
            response = r.post(url, data=args)

        except Exception as e:
            raise RuntimeError(f"API call failed: {e}")
        print(response.status_code)
        # check that the response has logprobs
        try:
            token_info = response.output[0].content[0]
            top_logprobs = token_info.logprobs
        except (AttributeError, IndexError):
            raise RuntimeError("Model response does not contain logprobs. Make sure your model supports logprobs.")

        # filter allowed tokens
        logprob_dict = {td.token: td.logprob for td in top_logprobs if td.token in allowed_tokens}

        if not logprob_dict:
            raise RuntimeError("No allowed tokens found in model logprobs.")

        # normalize probabilities
        tokens = list(logprob_dict.keys())
        logprobs = np.array([logprob_dict[t] for t in tokens])
        exp_probs = np.exp(logprobs - np.max(logprobs))
        probs = exp_probs / np.sum(exp_probs)

        return {t: float(p) for t, p in zip(tokens, probs)}

