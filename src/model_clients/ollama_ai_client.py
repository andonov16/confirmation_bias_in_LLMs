from typing import Dict
import numpy as np

from src.model_clients.base_model_client import BaseModelClient
from ollama import ChatResponse, Client


class OllamaAIClient(BaseModelClient):
    def __init__(self,
                 api_key: str,
                 base_url: str,
                 model_name: str):
        self.model_name = model_name
        self.base_url = base_url
        self.client = Client(host=self.base_url)

    def send_prompt_get_response(
        self,
        system_prompt: str,
        user_prompt: str,
        allowed_tokens: list[str] = None,
        temperature: float = 0.0,
    ) -> Dict[str, float]:
        raise NotImplemented()
