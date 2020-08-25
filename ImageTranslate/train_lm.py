import datetime
import os
import pickle
import sys
import time

import torch
import torch.nn as nn
import torch.utils.data as data_utils
from IPython.core import ultratb

from ImageTranslate import dataset
from ImageTranslate.lm import LM
from ImageTranslate.option_parser import get_lm_option_parser
from ImageTranslate.parallel import DataParallelModel, DataParallelCriterion
from ImageTranslate.reformer_lm import ReformerLM
from ImageTranslate.textprocessor import TextProcessor
from ImageTranslate.utils import build_optimizer, mask_text, unmask_text

sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)


class LMTrainer:
    def __init__(self, model, mask_prob: float = 0.15, clip: int = 1, optimizer=None):
        self.model = model
        self.clip = clip
        self.optimizer = optimizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.mask_prob = mask_prob
        self.criterion = nn.NLLLoss(ignore_index=model.text_processor.pad_token_id())

        num_gpu = torch.cuda.device_count()
        if num_gpu > 1:
            print("Let's use", num_gpu, "GPUs!")
            self.model = DataParallelModel(self.model)
            self.criterion = DataParallelCriterion(self.criterion)

        self.best_dev_loss = float("inf")
        self.best_train_loss = float("inf")
        self.last_train_loss = float("inf")

    def train_epoch(self, data_iter: data_utils.DataLoader, dev_data_iter: data_utils.DataLoader, saving_path: str,
                    step: int):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens, total_loss, tokens, cur_loss = 0, 0, 0, 0
        cur_loss = 0
        model = self.model.module if hasattr(self.model, "module") else self.model

        for i, batch in enumerate(data_iter):
            if self.optimizer is not None:
                self.optimizer.zero_grad()
            mask, target, texts = mask_text(self.mask_prob, batch["pad_mask"], batch["texts"], model.text_processor)
            try:
                predictions = self.model(mask=mask, texts=texts, pads=batch["pad_mask"], langs=batch["langs"])
                ntokens = target.size(0)

                if ntokens == 0:  # Nothing to predict!
                    continue

                loss = self.criterion(predictions, target).mean()
                loss.backward()

                unmask_text(mask, target, texts)

                if self.optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                    self.optimizer.step()
                    step += 1

                loss = float(loss.data) * ntokens
                total_loss += loss
                cur_loss += loss
                total_tokens += ntokens
                tokens += ntokens

                if step % 50 == 0:
                    elapsed = time.time() - start
                    print(datetime.datetime.now(),
                          "Epoch Step: %d Loss: %f Tokens per Sec: %f" % (step, cur_loss / tokens, tokens / elapsed))

                    if step % 500 == 0:
                        self.validate_and_save(saving_path, dev_data_iter)

                    start, tokens, cur_loss = time.time(), 0, 0
            except RuntimeError as err:
                print("Problem with batch item", texts.size())
                torch.cuda.empty_cache()
                pass

        current_loss = total_loss / total_tokens
        print("Total loss in this epoch: %f" % current_loss)
        if current_loss < self.best_train_loss:
            self.best_train_loss = current_loss
            model_to_save = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            model_to_save.save(saving_path + ".latest")
            with open(os.path.join(saving_path + ".latest", "optim"), "wb") as fp:
                pickle.dump(self.optimizer, fp)
        self.last_train_loss = current_loss

        self.validate_and_save(saving_path, dev_data_iter)
        return step

    def validate_and_save(self, saving_path, dev_data_iter):
        with torch.no_grad():
            model = self.model.module if hasattr(self.model, "module") else self.model
            model.eval()
            total_dev_loss, total_dev_tokens = 0, 0
            for batch in dev_data_iter:
                mask, target, texts = mask_text(self.mask_prob, batch["pad_mask"], batch["texts"].clone(),
                                                model.text_processor)
                predictions = self.model(mask=mask, texts=texts, pads=batch["pad_mask"], langs=batch["langs"])
                ntokens = target.size(0)

                if ntokens == 0:  # Nothing to predict!
                    continue
                loss = self.criterion(predictions, target).mean().data * ntokens
                total_dev_loss += float(loss)
                total_dev_tokens += ntokens

            dev_loss = total_dev_loss / total_dev_tokens
            print("Current dev loss", dev_loss)
            if self.best_dev_loss > float(dev_loss):
                self.best_dev_loss = float(dev_loss)
                print("saving best dev loss", self.best_dev_loss)
                model_to_save = (
                    self.model.module if hasattr(self.model, "module") else self.model
                )
                model_to_save.save(saving_path)
                with open(os.path.join(saving_path, "optim"), "wb") as fp:
                    pickle.dump(self.optimizer, fp)
            model.train()

    @staticmethod
    def config_dropout(model, dropout):
        model.encoder.config.hidden_dropout_prob = dropout
        model.encoder.config.attention_probs_dropout_prob = dropout

    @staticmethod
    def train(options):
        if not os.path.exists(options.model_path):
            os.makedirs(options.model_path)

        text_processor = TextProcessor(options.tokenizer_path)

        lm_class = ReformerLM if options.reformer else LM
        if options.pretrained_path is None:
            lm = lm_class(text_processor=text_processor, size=options.model_size)
        else:
            lm = lm_class.load(options.pretrained_path)

        if options.reformer:
            lm.config.hidden_dropout_prob = options.dropout
            lm.config.local_attention_probs_dropout_prob = options.dropout
            lm.config.lsh_attention_probs_dropout_prob = options.dropout
        else:
            LMTrainer.config_dropout(lm, options.dropout)

        train_data = dataset.TextDataset(save_cache_dir=options.train_path, max_cache_size=options.cache_size)
        dev_data = dataset.TextDataset(save_cache_dir=options.dev_path, max_cache_size=options.cache_size,
                                       load_all=True)

        if options.continue_train:
            with open(os.path.join(options.pretrained_path, "optim"), "rb") as fp:
                optimizer = pickle.load(fp)
        else:
            optimizer = build_optimizer(lm, options.learning_rate, options.warmup)

        trainer = LMTrainer(model=lm, mask_prob=options.mask_prob, optimizer=optimizer, clip=options.clip)

        collator = dataset.TextCollator(pad_idx=text_processor.pad_token_id())
        train_sampler, dev_sampler = None, None

        pin_memory = torch.cuda.is_available()
        loader = data_utils.DataLoader(train_data, batch_size=options.batch, shuffle=False, pin_memory=pin_memory,
                                       collate_fn=collator, sampler=train_sampler)
        dev_loader = data_utils.DataLoader(dev_data, batch_size=options.batch, shuffle=False, pin_memory=pin_memory,
                                           collate_fn=collator, sampler=dev_sampler)

        step, train_epoch = 0, 1
        while step <= options.step:
            print("train epoch", train_epoch)
            step = trainer.train_epoch(data_iter=loader, dev_data_iter=dev_loader, saving_path=options.model_path,
                                       step=step)


if __name__ == "__main__":
    parser = get_lm_option_parser()
    (options, args) = parser.parse_args()
    print(options)
    LMTrainer.train(options=options)
    print("Finished Training!")
