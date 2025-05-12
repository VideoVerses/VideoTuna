import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
import os
from ..utils import with_empty_init
from loguru import logger
class HunyuanClip(nn.Module):
    """
        Hunyuan clip code copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/hunyuandit/pipeline_hunyuandit.py
        hunyuan's clip used BertModel and BertTokenizer, so we copy it.
    """
    def __init__(self, model_dir, max_length=77, torch_dtype: torch.dtype = torch.bfloat16):
        super(HunyuanClip, self).__init__()
        
        self.model_dir = model_dir
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(model_dir, 'tokenizer'))
        self.config = BertConfig.from_pretrained(os.path.join(model_dir, 'clip_text_encoder'))
        self.text_encoder = BertModel(self.config)
        self.torch_dtype = torch_dtype

    def load_weight(self):
        #1. hunyuan has visual + bert, but we only want bert here, and remember remove bert prefix
        #2. BertModel has pooler layer, we do not need that
        logger.info("HunyuanClip: fixing bert model weights")
        state_dict = torch.load(os.path.join(self.model_dir, 'clip_text_encoder/pytorch_model.bin'), map_location='cpu')
        state_dict_pruned = {k[5:] : v for k, v in state_dict.items() if k.startswith('bert')}
        self.text_encoder.load_state_dict(state_dict_pruned, strict=False, assign=True)
        self.text_encoder = self.text_encoder.to(self.torch_dtype)
        
    @torch.no_grad
    def forward(self, prompts, with_mask=True, device='cuda'):
        self.device = device
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        prompt_embeds = self.text_encoder(
            text_inputs.input_ids.to(self.device),
            attention_mask=text_inputs.attention_mask.to(self.device) if with_mask else None,
        )
        return prompt_embeds.last_hidden_state, prompt_embeds.pooler_output
        