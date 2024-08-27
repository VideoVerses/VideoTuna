from abc import abstractmethod

class Base:
    def __init__(self, prompt_config, **kwargs) -> None:
        self.prompt_config = prompt_config

    @abstractmethod
    def complete(
        input_text, top_p=0.8, top_k=40, temperature=0.8, max_new_tokens=1024
    ):
        raise NotImplementedError()
