from typing import Dict, List


class BaseModelClient:
    def __init__(self,
                 api_key: str,
                 base_url: str,
                 model_name: str):
        raise NotImplemented()

    def send_prompt_get_response(self,
                                 system_prompt: str,
                                 user_prompt: str,
                                 allowed_tokens: List[str],
                                 temperature: float=0) -> Dict[str, float]:
        pass