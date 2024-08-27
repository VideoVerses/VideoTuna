from .base import Base
import lmdeploy


class CommonLLM(Base):
    def __init__(self, model_path, prompt_config, **kwargs) -> None:
        super().__init__(prompt_config, **kwargs)
        self.model_path = model_path
        self.model = lmdeploy.pipeline(model_path)

    def complete(
        self,
        input_text,
        top_p=0.8,
        top_k=40,
        temperature=0.8,
        max_new_tokens=1024,
    ):
        if isinstance(input_text, str):
            input_text = [
                input_text,
            ]

        real_input = []
        for i in range(len(input_text)):
            user_input = self.prompt_config.user_template.replace(
                "{{text_prompt}}", input_text[i].strip()
            )
            real_input.append(
                [
                    {"role": "system", "content": self.prompt_config.sys_template},
                    {"role": "user", "content": user_input},
                ]
            )
        response = self.model(
            real_input,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        return [r.text for r in response]
