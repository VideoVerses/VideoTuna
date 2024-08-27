import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from .base import Base


class DeepSeekMOE(Base):
    def __init__(self, model_path, prompt_config, **kwargs) -> None:
        super().__init__(prompt_config, **kwargs)
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).cuda()
        self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        self.model.generation_config.pad_token_id = (
            self.model.generation_config.eos_token_id
        )

    def complete(
        self,
        input_text,
        top_p=None,
        top_k=40,
        temperature=0.8,
        max_new_tokens=1024,
    ):
        if isinstance(input_text, str):
            input_text = [
                input_text,
            ]

        response = []
        for i in range(len(input_text)):
            user_input = self.prompt_config.user_template.replace(
                "{{text_prompt}}", input_text[i].strip()
            )
            real_input = [
                {"role": "system", "content": self.prompt_config.sys_template},
                {"role": "user", "content": user_input},
            ]
            input_tensor = self.tokenizer.apply_chat_template(
                real_input, add_generation_prompt=True, return_tensors="pt"
            )
            # outputs = model.generate(input_tensor.to(model.device), max_new_tokens=100)
            outputs = self.model.generate(
                input_tensor.to(self.model.device),
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            result = self.tokenizer.decode(
                outputs[0][input_tensor.shape[1] :], skip_special_tokens=True
            )
            response.append(result)

        return response
