import datetime
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from itertools import chain
from typing import List

import sacrebleu
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from IPython.core import ultratb

try:
    from apex import amp
except:
    pass
from torch.nn.utils.rnn import pad_sequence

from ImageTranslate import dataset
from ImageTranslate.image_model import ImageMassSeq2Seq
from ImageTranslate.lm import LM
from ImageTranslate.loss import SmoothedNLLLoss
from ImageTranslate.option_parser import get_img_options_parser
from ImageTranslate.parallel import DataParallelModel, DataParallelCriterion
from ImageTranslate.seq2seq import Seq2Seq
from ImageTranslate.seq_gen import BeamDecoder, get_outputs_until_eos
from ImageTranslate.textprocessor import TextProcessor
from ImageTranslate.utils import build_optimizer, mass_mask, mass_unmask, backward

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


def get_lex_dict(dict_path):
    lex_dict = defaultdict(list)
    with open(dict_path) as dr:
        for line in dr:
            elements = list(map(lambda x: int(x), line.strip().split(" ")))
            for element in elements[1:]:
                lex_dict[elements[0]].append(element)
    return lex_dict


class ImageMTTrainer:
    def __init__(self, model, mask_prob: float = 0.3, clip: int = 1, optimizer=None,
                 beam_width: int = 5, max_len_a: float = 1.1, max_len_b: int = 5,
                 len_penalty_ratio: float = 0.8, nll_loss: bool = False, fp16: bool = False, mm_mode="mixed"):
        self.model = model

        self.clip = clip
        self.optimizer = optimizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.num_gpu = torch.cuda.device_count()

        self.mask_prob = mask_prob
        if nll_loss:
            self.criterion = nn.NLLLoss(ignore_index=model.text_processor.pad_token_id())
        else:
            self.criterion = SmoothedNLLLoss(ignore_index=model.text_processor.pad_token_id())

        self.fp16 = False
        if self.num_gpu == 1 and fp16:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O2")
            self.fp16 = True
        self.generator = BeamDecoder(self.model, beam_width=beam_width, max_len_a=max_len_a, max_len_b=max_len_b,
                                     len_penalty_ratio=len_penalty_ratio)
        if self.num_gpu > 1:
            print("Let's use", self.num_gpu, "GPUs!")
            self.model = DataParallelModel(self.model)
            self.criterion = DataParallelCriterion(self.criterion)
            self.generator = DataParallelModel(self.generator)

        self.reference = None
        self.best_bleu = -1.0
        self.mm_mode = mm_mode

    def train_epoch(self, img_data_iter: List[data_utils.DataLoader] = None, step: int = 10, saving_path: str = None,
                    mass_data_iter: List[data_utils.DataLoader] = None, mt_dev_iter: List[data_utils.DataLoader] = None,
                    mt_train_iter: List[data_utils.DataLoader] = None, max_step: int = 300000,
                    fine_tune: bool = False, lang_directions: dict = False, lex_dict=None, save_opt: bool = False,
                    **kwargs):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens, cur_loss = 0, 0, 0, 0
        cur_loss = 0
        batch_zip, shortest = self.get_batch_zip(img_data_iter, mass_data_iter, mt_train_iter)

        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        for i, batches in enumerate(batch_zip):
            for batch in batches:
                self.optimizer.zero_grad()
                is_img_batch = isinstance(batch, list) and "captions" in batch[0]
                is_mass_batch = not is_img_batch and "dst_texts" not in batch
                is_contrastive = False
                try:

                    src_inputs = batch["src_texts"].squeeze(0)
                    src_mask = batch["src_pad_mask"].squeeze(0)
                    tgt_inputs = batch["dst_texts"].squeeze(0)
                    tgt_mask = batch["dst_pad_mask"].squeeze(0)
                    src_langs = batch["src_langs"].squeeze(0)
                    dst_langs = batch["dst_langs"].squeeze(0)
                    proposals = batch["proposal"].squeeze(0) if lex_dict is not None else None
                    if src_inputs.size(0) < self.num_gpu:
                        continue
                    predictions = self.model(src_inputs=src_inputs, tgt_inputs=tgt_inputs,
                                             src_pads=src_mask, tgt_mask=tgt_mask, src_langs=src_langs,
                                             tgt_langs=dst_langs, proposals=proposals,
                                             pad_idx=model.text_processor.pad_token_id(), log_softmax=True)
                    targets = tgt_inputs[:, 1:].contiguous().view(-1)
                    tgt_mask_flat = tgt_mask[:, 1:].contiguous().view(-1)
                    targets = targets[tgt_mask_flat]
                    ntokens = targets.size(0)


                    if self.num_gpu == 1:
                        targets = targets.to(predictions.device)

                    loss = self.criterion(predictions, targets).mean()
                    backward(loss, self.optimizer, self.fp16)

                    loss = float(loss.data) * ntokens
                    tokens += ntokens
                    total_tokens += ntokens
                    total_loss += loss
                    cur_loss += loss

                    # We accumulate the gradients for both tasks!
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                    self.optimizer.step()
                    step += 1


                    if step % 50 == 0 and tokens > 0:
                        elapsed = time.time() - start
                        print(datetime.datetime.now(),
                              "Epoch Step: %d Loss: %f Tokens per Sec: %f " % (
                                  step, cur_loss / tokens, tokens / elapsed))

                        if step % 500 == 0:
                            if mt_dev_iter is not None and step % 5000 == 0:
                                bleu = self.eval_bleu(mt_dev_iter, saving_path)
                                print("BLEU:", bleu)

                            model.save(saving_path)
                            if save_opt:
                                with open(os.path.join(saving_path, "optim"), "wb") as fp:
                                    pickle.dump(self.optimizer, fp)

                        start, tokens, cur_loss = time.time(), 0, 0

                except RuntimeError as err:
                    print(repr(err))
                    print("Error processing", is_img_batch)
                    if (isinstance(model, ImageMassSeq2Seq)) and is_img_batch:
                        for b in batch:
                            print("->", len(b["images"]), b["captions"].size())
                    torch.cuda.empty_cache()

            if i == shortest - 1:
                break
            if step >= max_step:
                break

        try:
            print("Total loss in this epoch: %f" % (total_loss / total_tokens))
            model.save(saving_path)

            if mt_dev_iter is not None:
                bleu = self.eval_bleu(mt_dev_iter, saving_path)
                print("BLEU:", bleu)
        except RuntimeError as err:
            print(repr(err))

        return step

    def get_batch_zip(self, img_data_iter, mass_data_iter, mt_train_iter):
        # if img_data_iter is not None and mt_train_iter is not None:
        #     img_data_iter *= 5
        # if mass_data_iter is not None and mt_train_iter is not None:
        #     mass_data_iter *= 5
        iters = list(chain(*filter(lambda x: x != None, [img_data_iter, mass_data_iter, mt_train_iter])))
        shortest = min(len(l) for l in iters)
        return zip(*iters), shortest

    def eval_bleu(self, dev_data_iter, saving_path, save_opt: bool = False):
        mt_output = []
        src_text = []
        model = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model.eval()

        with torch.no_grad():
            for iter in dev_data_iter:
                for batch in iter:
                    src_inputs = batch["src_texts"].squeeze(0)
                    src_mask = batch["src_pad_mask"].squeeze(0)
                    tgt_inputs = batch["dst_texts"].squeeze(0)
                    src_langs = batch["src_langs"].squeeze(0)
                    dst_langs = batch["dst_langs"].squeeze(0)
                    src_pad_idx = batch["pad_idx"].squeeze(0)
                    proposal = batch["proposal"].squeeze(0) if batch["proposal"] is not None else None

                    src_ids = get_outputs_until_eos(model.text_processor.sep_token_id(), src_inputs,
                                                    remove_first_token=True)
                    src_text += list(map(lambda src: model.text_processor.tokenizer.decode(src.numpy()), src_ids))

                    outputs = self.generator(src_inputs=src_inputs, src_sizes=src_pad_idx,
                                             first_tokens=tgt_inputs[:, 0],
                                             src_mask=src_mask, src_langs=src_langs, tgt_langs=dst_langs,
                                             pad_idx=model.text_processor.pad_token_id(), proposals=proposal)
                    if self.num_gpu > 1:
                        new_outputs = []
                        for output in outputs:
                            new_outputs += output
                        outputs = new_outputs

                    mt_output += list(map(lambda x: model.text_processor.tokenizer.decode(x[1:].numpy()), outputs))

            model.train()
        bleu = sacrebleu.corpus_bleu(mt_output, [self.reference[:len(mt_output)]], lowercase=True, tokenize="intl")

        with open(os.path.join(saving_path, "bleu.output"), "w") as writer:
            writer.write("\n".join(
                [src + "\n" + ref + "\n" + o + "\n\n***************\n" for src, ref, o in
                 zip(src_text, mt_output, self.reference[:len(mt_output)])]))

        if bleu.score > self.best_bleu:
            self.best_bleu = bleu.score
            print("Saving best BLEU", self.best_bleu)
            with open(os.path.join(saving_path, "bleu.best.output"), "w") as writer:
                writer.write("\n".join(
                    [src + "\n" + ref + "\n" + o + "\n\n***************\n" for src, ref, o in
                     zip(src_text, mt_output, self.reference[:len(mt_output)])]))

            model.save(saving_path)
            if save_opt:
                with open(os.path.join(saving_path, "optim"), "wb") as fp:
                    pickle.dump(self.optimizer, fp)

        return bleu.score

    @staticmethod
    def train(options):
        lex_dict = None
        if options.dict_path is not None:
            lex_dict = get_lex_dict(options.dict_path)
        if not os.path.exists(options.model_path):
            os.makedirs(options.model_path)

        text_processor = TextProcessor(options.tokenizer_path)
        assert text_processor.pad_token_id() == 0
        num_processors = max(torch.cuda.device_count(), 1)

        if options.pretrained_path is not None:
            print("Loading pretrained path", options.pretrained_path)
            mt_model = Seq2Seq.load(ImageMassSeq2Seq, options.pretrained_path, tok_dir=options.tokenizer_path)
        else:
            mt_model = ImageMassSeq2Seq(use_proposals=lex_dict is not None, tie_embed=options.tie_embed,
                                        text_processor=text_processor, resnet_depth=options.resnet_depth,
                                        lang_dec=options.lang_decoder, enc_layer=options.encoder_layer,
                                        dec_layer=options.decoder_layer, embed_dim=options.embed_dim,
                                        intermediate_dim=options.intermediate_layer_dim)

        if options.lm_path is not None:
            lm = LM(text_processor=text_processor, enc_layer=options.encoder_layer,
                    embed_dim=options.embed_dim, intermediate_dim=options.intermediate_layer_dim)
            mt_model.init_from_lm(lm)

        print("Model initialization done!")

        # We assume that the collator function returns a list with the size of number of gpus (in case of cpus,
        collator = dataset.ImageTextCollator()
        num_batches = max(1, torch.cuda.device_count())

        if options.continue_train:
            with open(os.path.join(options.pretrained_path, "optim"), "rb") as fp:
                optimizer = pickle.load(fp)
        else:
            optimizer = build_optimizer(mt_model, options.learning_rate, warump_steps=options.warmup)
        trainer = ImageMTTrainer(model=mt_model, mask_prob=options.mask_prob, optimizer=optimizer, clip=options.clip,
                                 beam_width=options.beam_width, max_len_a=options.max_len_a,
                                 max_len_b=options.max_len_b, len_penalty_ratio=options.len_penalty_ratio,
                                 fp16=options.fp16, mm_mode=options.mm_mode)

        pin_memory = torch.cuda.is_available()

        mt_train_loader = None
        if options.mt_train_path is not None:
            mt_train_loader = ImageMTTrainer.get_mt_train_data(mt_model, num_processors, options, pin_memory,
                                                               lex_dict=lex_dict)

        mt_dev_loader = None
        if options.mt_dev_path is not None:
            mt_dev_loader = ImageMTTrainer.get_mt_dev_data(mt_model, options, pin_memory, text_processor, trainer,
                                                           lex_dict=lex_dict)

        step, train_epoch = 0, 1
        while options.step > 0 and step < options.step and train_epoch <= 10:
            print("train epoch", train_epoch, "step:", step)
            step = trainer.train_epoch(mt_train_iter=mt_train_loader, max_step=options.step, lex_dict=lex_dict,
                                       mt_dev_iter=mt_dev_loader, saving_path=options.model_path, step=step,
                                       save_opt=False)
            train_epoch += 1

    @staticmethod
    def get_mt_dev_data(mt_model, options, pin_memory, text_processor, trainer, lex_dict=None):
        mt_dev_loader = []
        dev_paths = options.mt_dev_path.split(",")
        trainer.reference = []
        for dev_path in dev_paths:
            mt_dev_data = dataset.MTDataset(batch_pickle_dir=dev_path,
                                            max_batch_capacity=options.total_capacity, keep_pad_idx=True,
                                            max_batch=int(options.batch / (options.beam_width * 2)),
                                            pad_idx=mt_model.text_processor.pad_token_id(), lex_dict=lex_dict)
            dl = data_utils.DataLoader(mt_dev_data, batch_size=1, shuffle=False, pin_memory=pin_memory)
            mt_dev_loader.append(dl)

            print("creating reference")

            generator = (
                trainer.generator.module if hasattr(trainer.generator, "module") else trainer.generator
            )

            for batch in dl:
                tgt_inputs = batch["dst_texts"].squeeze()
                refs = get_outputs_until_eos(text_processor.sep_token_id(), tgt_inputs, remove_first_token=True)
                ref = [generator.seq2seq_model.text_processor.tokenizer.decode(ref.numpy()) for ref in refs]
                trainer.reference += ref
        return mt_dev_loader

    @staticmethod
    def get_mt_train_data(mt_model, num_processors, options, pin_memory, lex_dict=None):
        mt_train_loader = []
        train_paths = options.mt_train_path.split(",")
        for train_path in train_paths:
            mt_train_data = dataset.MTDataset(batch_pickle_dir=train_path,
                                              max_batch_capacity=int(num_processors * options.total_capacity / 2),
                                              max_batch=int(num_processors * options.batch / 2),
                                              pad_idx=mt_model.text_processor.pad_token_id(), lex_dict=lex_dict,
                                              keep_pad_idx=False)
            mtl = data_utils.DataLoader(mt_train_data, batch_size=1, shuffle=True, pin_memory=pin_memory)
            mt_train_loader.append(mtl)
        return mt_train_loader


if __name__ == "__main__":
    parser = get_img_options_parser()
    (options, args) = parser.parse_args()
    print(options)
    ImageMTTrainer.train(options=options)
    print("Finished Training!")
