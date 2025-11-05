from openai import OpenAI, OpenAIError

from src.model_clients.base_model_client import BaseModelClient


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

    def send_prompt_get_response(self,
                                 system_prompt: str,
                                 user_prompt: str,
                                 temperature: float=0) -> str:
        """
        Sends the prompt to OpenAI's API and returns the generated response.
        """
        try:
            response = self.client.responses.create(
                model=self.model_name,
                instructions=system_prompt,
                input=user_prompt,
                reasoning={"effort": "minimal"},
            )
        except OpenAIError as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")

        return response.output[1].content[0].text.strip()