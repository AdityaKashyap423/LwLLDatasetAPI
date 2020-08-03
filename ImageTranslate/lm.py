import os
import pickle
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

import lm_config
from bert_seq2seq import BertOutputLayer
from textprocessor import TextProcessor


class LM(nn.Module):
    def __init__(self, text_processor: TextProcessor, config: BertConfig = None, encoder: BertModel = None,
                 enc_layer: int = 6, embed_dim: int = 768, intermediate_dim: int = 3072):
        super(LM, self).__init__()
        self.text_processor: TextProcessor = text_processor

        if config is not None:
            self.config = config
        else:
            self.config = lm_config.get_config(vocab_size=text_processor.tokenizer.get_vocab_size(),
                                               pad_token_id=text_processor.pad_token_id(),
                                               bos_token_id=text_processor.bos_token_id(),
                                               eos_token_id=text_processor.sep_token_id(),
                                               enc_layer=enc_layer, embed_dim=embed_dim,
                                               intermediate_dim=intermediate_dim)

            self.config["type_vocab_size"] = len(text_processor.languages)
            self.config = BertConfig(**self.config)

        self.masked_lm = BertOutputLayer(self.config)
        if encoder is None:
            self.encoder: BertModel = BertModel(self.config)
            self.encoder.init_weights()
        else:
            self.encoder = encoder
        self.encoder._tie_or_clone_weights(self.masked_lm.decoder, self.encoder.embeddings.word_embeddings)

    def forward(self, mask: torch.Tensor, texts: torch.Tensor, pads: torch.Tensor, langs: List = None):
        """
        :param data: A minibatch as dictionary that has transformed image and tokenized text as long tensors.
        :return:
        """
        langs_tensor = langs.squeeze().unsqueeze(1).expand(-1, texts.size(1))

        device = self.encoder.embeddings.word_embeddings.weight.device
        texts = texts.to(device)
        pads = pads.to(device)
        langs_tensor = langs_tensor.to(device)
        text_hidden, text_cls_head = self.encoder(texts, attention_mask=pads, token_type_ids=langs_tensor)
        output_predictions = F.log_softmax(self.masked_lm(text_hidden[mask]), dim=1)
        return output_predictions

    def save(self, out_dir: str):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(os.path.join(out_dir, "config"), "wb") as fp:
            pickle.dump(self.config, fp)

        torch.save(self.state_dict(), os.path.join(out_dir, "model.state_dict"))
        self.text_processor.save(directory=out_dir)

    @staticmethod
    def load(out_dir: str):
        text_processor = TextProcessor(tok_model_path=out_dir)
        with open(os.path.join(out_dir, "config"), "rb") as fp:
            config = pickle.load(fp)
            if isinstance(config, dict):
                # For older configs
                config = BertConfig(**config)
            lm = LM(text_processor=text_processor, config=config)
            lm.load_state_dict(torch.load(os.path.join(out_dir, "model.state_dict")))
            return lm
