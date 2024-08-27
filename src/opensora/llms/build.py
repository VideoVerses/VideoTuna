from .common_llm import CommonLLM
from .deepseek_moe import DeepSeekMOE

COMMON_LLMS = [
    "internlm",
    "llama",
    "qwen",
    "baichuan",
    "deepseek",
]


def build(model_name, model_path, prompt_config, **kwargs):
    if model_name in COMMON_LLMS:
        return CommonLLM(model_path, prompt_config, **kwargs)
    elif model_name == "openai":
        pass
    elif model_name == "deepseek_moe":
        return DeepSeekMOE(model_path, prompt_config, **kwargs)
    else:
        raise NotImplementedError()

 