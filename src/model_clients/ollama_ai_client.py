from typing import Dict
import numpy as np
import requests as r
import json

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
            "system": system_prompt,
            "prompt": user_prompt,
            "logprobs": True,
            "top_logprobs": 20,
            "stream:": False,
            "options": {
                "temperature": 0,
                "num_predict": 5
            }
        }
        try:
            response = r.post(url, json=args)

        except Exception as e:
            raise RuntimeError(f"API call failed: {e}")
        print(response.text)
        try:
            logprobs_list = response.json().get(logprobs)
            top_logprobs = {lp.token: lp.logprob for lp in logprobs_list}
            print(top_logprobs)
        except (AttributeError, IndexError):
            raise RuntimeError("Model response does not contain logprobs. Make sure your model supports logprobs.")

        # filter allowed tokens
        logprob_dict = {td_key: td_value for td_key, td_value in top_logprobs.items() if td_key in allowed_tokens}

        if not logprob_dict:
            raise RuntimeError("No allowed tokens found in model logprobs.")

        # normalize probabilities
        tokens = list(logprob_dict.keys())
        logprobs = np.array([logprob_dict[t] for t in tokens])
        exp_probs = np.exp(logprobs - np.max(logprobs))
        probs = exp_probs / np.sum(exp_probs)

        return {t: float(p) for t, p in zip(tokens, probs)}

