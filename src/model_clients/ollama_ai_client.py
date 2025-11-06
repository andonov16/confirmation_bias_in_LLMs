from sklearn import base
from src.model_clients.base_model_client import BaseModelClient
from ollama import ChatResponse, Client

class OllamaAIClient(BaseModelClient):
    def __init__(self, api_key: str, base_url: str, model_name: str):
        """
        Initialize the client with your Ollama API key.
        """
        self.model_name = model_name
        self.base_url = base_url
        
        self.client = Client(host=self.base_url)
    
    def send_prompt_get_response(self, system_prompt: str, user_prompt: str, temperature: float=0) -> str:
        """
        Sends the prompt to Ollama's API and returns the generated response.
        """
        try:
            response: ChatResponse = self.client.chat(model=self.model_name, messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
        except Exception as e:
            raise RuntimeError(f"Ollama API call failed: {e}")
        
        return response.message.content.strip()