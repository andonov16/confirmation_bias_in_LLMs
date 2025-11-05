class BaseModelClient:
    def __init__(self,):
        raise NotImplemented()

    def send_prompt_get_response(self,
                                 system_prompt: str,
                                 user_prompt: str) -> str:
        raise NotImplemented()
