from typing import Any, Dict
from torch import nn


class TextEmbExtractor(nn.Module):
    def __init__(self, tokenizer, text_encoder) -> None:
        super(TextEmbExtractor, self).__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

    def forward(
        self,
        texts,
        text_params: Dict = None,
    ):
        if text_params is None:
            text_params = {}
        special_prompt_input = self.tokenizer(
            texts,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = special_prompt_input.attention_mask.to(
                self.text_encoder.device
            )
        else:
            attention_mask = None

        embeddings = self.text_encoder(
            special_prompt_input.input_ids.to(self.text_encoder.device),
            attention_mask=attention_mask,
            **text_params
        )
        return embeddings
