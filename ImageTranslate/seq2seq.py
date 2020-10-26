import copy
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from ImageTranslate import lm_config
from ImageTranslate.bert_seq2seq import BertEncoderModel, BertDecoderModel, BertOutputLayer, BertConfig
from ImageTranslate.lm import LM
from ImageTranslate.textprocessor import TextProcessor


def future_mask(tgt_mask):
    attn_shape = (tgt_mask.size(0), tgt_mask.size(1), tgt_mask.size(1))
    future_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type_as(tgt_mask)
    return ~future_mask & tgt_mask.unsqueeze(-1)


class Seq2Seq(nn.Module):
    def __init__(self, text_processor: TextProcessor, lang_dec: bool = True, use_proposals=False, tie_embed=False,
                 enc_layer: int = 6, dec_layer: int = 3, embed_dim: int = 768, intermediate_dim: int = 3072,
                 freeze_image: bool = False, resnet_depth: int = 1):
        super(Seq2Seq, self).__init__()
        self.text_processor: TextProcessor = text_processor
        self.config = lm_config.get_config(vocab_size=text_processor.tokenizer.get_vocab_size(),
                                           pad_token_id=text_processor.pad_token_id(),
                                           bos_token_id=text_processor.bos_token_id(),
                                           eos_token_id=text_processor.sep_token_id(),
                                           enc_layer=enc_layer, embed_dim=embed_dim, intermediate_dim=intermediate_dim)

        self.enc_layer = enc_layer
        self.dec_layer = dec_layer
        self.embed_dim = embed_dim
        self.intermediate_dim = intermediate_dim
        self.config["type_vocab_size"] = len(text_processor.languages)
        self.config = BertConfig(**self.config)
        dec_config = copy.deepcopy(self.config)
        dec_config.num_hidden_layers = self.dec_layer

        self.encoder = BertEncoderModel(self.config)
        self.encoder.init_weights()
        self.lang_dec = lang_dec
        self.tie_embed = tie_embed
        if not lang_dec:
            self.decoder = BertDecoderModel(dec_config)
            self.encoder._tie_or_clone_weights(self.encoder.embeddings.position_embeddings,
                                               self.decoder.embeddings.position_embeddings)
            self.encoder._tie_or_clone_weights(self.encoder.embeddings.token_type_embeddings,
                                               self.decoder.embeddings.token_type_embeddings)
            self.encoder._tie_or_clone_weights(self.encoder.embeddings.word_embeddings,
                                               self.decoder.embeddings.word_embeddings)

            if tie_embed:
                self.output_layer = BertOutputLayer(dec_config)
                self.encoder._tie_or_clone_weights(self.output_layer, self.encoder.embeddings.word_embeddings)
                self.encoder._tie_or_clone_weights(self.encoder.embeddings.position_embeddings,
                                                   self.decoder.embeddings.position_embeddings)
                self.decoder._tie_or_clone_weights(self.output_layer, self.decoder.embeddings.word_embeddings)
            else:
                self.output_layer = nn.ModuleList([BertOutputLayer(dec_config) for _ in text_processor.languages])

            if len(self.encoder.encoder.layer) == len(self.decoder.decoder.layer):
                for i in range(len(self.encoder.encoder.layer)):
                    self.decoder.decoder.layer[i].attention = self.encoder.encoder.layer[i].attention

        else:
            dec = BertDecoderModel(dec_config)
            self.decoder = nn.ModuleList([copy.deepcopy(dec) for _ in text_processor.languages])
            self.output_layer = nn.ModuleList([BertOutputLayer(dec_config) for _ in text_processor.languages])
            for i, dec in enumerate(self.decoder):
                if tie_embed:
                    self.encoder._tie_or_clone_weights(self.output_layer[i], self.encoder.embeddings.word_embeddings)
                    dec.embeddings.position_embeddings = self.encoder.embeddings.position_embeddings
                dec._tie_or_clone_weights(self.output_layer[i], dec.embeddings.word_embeddings)
                dec._tie_or_clone_weights(self.encoder.embeddings.token_type_embeddings,
                                          dec.embeddings.token_type_embeddings)

        self.use_proposals = use_proposals
        if self.use_proposals:
            self.proposal_embedding = self.encoder.embeddings.word_embeddings
            self.lexical_gate = nn.Parameter(torch.zeros(1, self.config.hidden_size).fill_(0.1), requires_grad=True)
            self.lexical_layer_norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)

        self.freeze_image = freeze_image
        self.resnet_depth = resnet_depth

    def init_from_lm(self, lm: LM):
        self.encoder = lm.encoder
        if not self.lang_dec:
            self.output_layer = lm.masked_lm
            self.decoder = BertDecoderModel(self.config)
            self.decoder._tie_or_clone_weights(self.output_layer.decoder, self.decoder.embeddings.word_embeddings)
        else:
            dec = self.decoder_cls(self.config)
            self.decoder = nn.ModuleList([copy.deepcopy(dec) for _ in self.text_processor.languages])
            self.output_layer = nn.ModuleList([copy.deepcopy(lm.masked_lm) for _ in self.text_processor.languages])
            for i, dec in enumerate(self.decoder):
                dec._tie_or_clone_weights(self.output_layer[i].decoder, dec.embeddings.word_embeddings)

    def encode(self, src_inputs, src_mask, src_langs, images=None):
        device = self.encoder.embeddings.word_embeddings.weight.device
        if src_inputs.device != device:
            src_inputs = src_inputs.to(device)
            src_mask = src_mask.to(device)
            src_langs = src_langs.to(device)
        encoder_states = self.encoder(src_inputs, attention_mask=src_mask, token_type_ids=src_langs)
        return (encoder_states, None)

    def attend_proposal(self, decoder_output, proposals, pad_idx):
        device = self.encoder.embeddings.word_embeddings.weight.device
        proposals = proposals.to(device)
        attend_mask = (proposals == pad_idx)
        mapped_output = decoder_output
        proposal_embedding = self.proposal_embedding(proposals)
        if decoder_output.dim() == 3:
            proposal_embedding = proposal_embedding.unsqueeze(1)
            proposal_embedding = proposal_embedding.expand(-1, decoder_output.size(1), -1, -1)
            mapped_output = mapped_output.unsqueeze(2)
            attend_mask = attend_mask.unsqueeze(1)
        else:
            if len(proposals) < len(decoder_output):
                beam_width = int(len(decoder_output) / len(proposals))
                proposals = torch.repeat_interleave(proposals, beam_width, 0)
                attend_mask = torch.repeat_interleave(attend_mask, beam_width, 0)
                proposal_embedding = torch.repeat_interleave(proposal_embedding, beam_width, 0)
            mapped_output = mapped_output.unsqueeze(1)

        attend_scores = torch.matmul(mapped_output, proposal_embedding.transpose(-1, -2)).squeeze(-2)

        attend_mask = attend_mask.expand(attend_scores.size())
        attend_scores[attend_mask].fill_(-10000.0)
        attend_probs = nn.Softmax(dim=-1)(attend_scores)

        proposal_context = torch.sum(attend_probs.unsqueeze(-1) * proposal_embedding, dim=-2)
        proposal_values = proposal_context
        final_proposal_mask = torch.all(proposals == pad_idx, dim=-1)
        proposal_values[final_proposal_mask] = 1e-8

        sig_gate = torch.sigmoid(self.lexical_gate + 1e-8)

        combined_value = sig_gate * decoder_output + (1 - sig_gate) * proposal_values
        combined_value = self.lexical_layer_norm(combined_value)
        return combined_value

    def forward(self, src_inputs, tgt_inputs, src_mask, tgt_mask, src_langs, tgt_langs, proposals=None,
                log_softmax: bool = False):
        "Take in and process masked src and target sequences."
        device = self.encoder.embeddings.word_embeddings.weight.device
        batch_lang = int(tgt_langs[0])
        src_langs = src_langs.unsqueeze(-1).expand(-1, src_inputs.size(-1))
        tgt_langs = tgt_langs.unsqueeze(-1).expand(-1, tgt_inputs.size(-1)).to(device)
        src_inputs = src_inputs.to(device)
        src_langs = src_langs.to(device)
        if tgt_inputs.device != device:
            tgt_inputs = tgt_inputs.to(device)
            tgt_mask = tgt_mask.to(device)
        if src_mask.device != device:
            src_mask = src_mask.to(device)

        encoder_states = self.encode(src_inputs, src_mask, src_langs)[0]

        subseq_mask = future_mask(tgt_mask[:, :-1])
        if subseq_mask.device != tgt_inputs.device:
            subseq_mask = subseq_mask.to(device)

        decoder = self.decoder if not self.lang_dec else self.decoder[batch_lang]
        output_layer = self.output_layer if (not self.lang_dec) and self.tie_embed else self.output_layer[batch_lang]

        decoder_output = decoder(encoder_states=encoder_states, input_ids=tgt_inputs[:, :-1],
                                 encoder_attention_mask=src_mask, tgt_attention_mask=subseq_mask,
                                 token_type_ids=tgt_langs[:, :-1])
        if self.use_proposals:
            decoder_output = self.attend_proposal(decoder_output, proposals, self.text_processor.pad_token_id())
        diag_outputs_flat = decoder_output.view(-1, decoder_output.size(-1))
        tgt_non_mask_flat = tgt_mask[:, 1:].contiguous().view(-1)
        non_padded_outputs = diag_outputs_flat[tgt_non_mask_flat]
        outputs = output_layer(non_padded_outputs)
        if log_softmax:
            outputs = F.log_softmax(outputs, dim=-1)
        return outputs

    def save(self, out_dir: str):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(os.path.join(out_dir, "mt_config"), "wb") as fp:
            pickle.dump(
                (self.lang_dec, self.use_proposals, self.enc_layer, self.dec_layer, self.embed_dim,
                 self.intermediate_dim, self.tie_embed, self.resnet_depth, self.freeze_image), fp)
        try:
            torch.save(self.state_dict(), os.path.join(out_dir, "mt_model.state_dict"))
        except:
            torch.cuda.empty_cache()
            torch.save(self.state_dict(), os.path.join(out_dir, "mt_model.state_dict"))
        finally:
            torch.cuda.empty_cache()

    @staticmethod
    def load(cls, out_dir: str, tok_dir: str):
        text_processor = TextProcessor(tok_model_path=tok_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(out_dir, "mt_config"), "rb") as fp:
            lang_dec, use_proposals, enc_layer, dec_layer, embed_dim, intermediate_dim, tie_embed, resnet_depth, freeze_image = pickle.load(
                fp)

            print("TRANSLATE OPTIONS", lang_dec, use_proposals, enc_layer, dec_layer, embed_dim, intermediate_dim,
                  tie_embed, resnet_depth, freeze_image)
            mt_model = cls(text_processor=text_processor, lang_dec=lang_dec, use_proposals=use_proposals,
                           tie_embed=tie_embed, enc_layer=enc_layer, dec_layer=dec_layer, embed_dim=embed_dim,
                           intermediate_dim=intermediate_dim, freeze_image=freeze_image, resnet_depth=resnet_depth)
            mt_model.load_state_dict(torch.load(os.path.join(out_dir, "mt_model.state_dict"), map_location=device),
                                     strict=False)
            return mt_model
