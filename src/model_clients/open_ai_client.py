from openai import OpenAI, OpenAIError
from typing import List, Dict
import numpy as np

from src.model_clients.base_model_client import BaseModelClient
from src.utils.tokens import get_logprobs_dict


class OpenAIClient(BaseModelClient):
    def __init__(self,
                 api_key: str,
                 base_url: str,
                 model_name: str):
        """
        Initialize the client with your OpenAI API key.
        """
        self.model_name = model_name
        self.base_url = base_url

        self.client = OpenAI(api_key=api_key, base_url=self.base_url)


    def send_prompt_get_response(
            self,
            system_prompt: str,
            user_prompt: str,
            allowed_tokens: List[str],
            temperature: float = 0.0,
    ) -> Dict[str, float]:
        #logit_bias = _get_logit_bias(model_name=self.model_name, allowed_tokens=allowed_tokens)

        try:
            response = self.client.responses.create(
                model=self.model_name,
                input= system_prompt + "\n\n" + user_prompt,
                # instructions=system_prompt,
                include=["message.output_text.logprobs"],  # request token probabilities
                temperature=temperature,
                top_logprobs=20,
                max_output_tokens= 160,
                reasoning={
                    "effort": "none"
                }
            )

        except OpenAIError as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")

        # check that the response has logprobs
        try:
            logprobs_list = response.output[0].content[0].logprobs
            top_logprobs = get_logprobs_dict(logprobs_list)
            print(top_logprobs)
        except (AttributeError, IndexError):
            raise RuntimeError("Model response does not contain logprobs. Make sure your model supports logprobs.")

        # filter allowed tokens
        logprob_dict = {td_key.strip(): td_value for td_key, td_value in top_logprobs.items() if td_key.strip() in allowed_tokens}

        if not logprob_dict:
            raise RuntimeError("No allowed tokens found in model logprobs.")

        # normalize probabilities
        tokens = list(logprob_dict.keys())
        logprobs = np.array([logprob_dict[t] for t in tokens])
        exp_probs = np.exp(logprobs - np.max(logprobs))
        probs = exp_probs / np.sum(exp_probs)

        return {t: float(p) for t, p in zip(tokens, probs)}
