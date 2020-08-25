import datetime
from optparse import OptionParser

import torch
import torch.utils.data as data_utils

try:
    from apex import amp
except:
    pass

from ImageTranslate import dataset
from ImageTranslate.parallel import DataParallelModel
from ImageTranslate.seq2seq import Seq2Seq
from ImageTranslate.seq_gen import BeamDecoder, get_outputs_until_eos


def get_lm_option_parser():
    parser = OptionParser()
    parser.add_option("--input", dest="input_path", metavar="FILE", default=None)
    parser.add_option("--src", dest="src_lang", type="str", default=None)
    parser.add_option("--target", dest="target_lang", type="str", default=None)
    parser.add_option("--output", dest="output_path", metavar="FILE", default=None)
    parser.add_option("--batch", dest="batch", help="Batch size", type="int", default=512)
    parser.add_option("--tok", dest="tokenizer_path", help="Path to the tokenizer folder", metavar="FILE", default=None)
    parser.add_option("--cache_size", dest="cache_size", help="Number of blocks in cache", type="int", default=300)
    parser.add_option("--model", dest="model_path", metavar="FILE", default=None)
    parser.add_option("--verbose", action="store_true", dest="verbose", help="Include input!", default=False)
    parser.add_option("--beam", dest="beam_width", type="int", default=4)
    parser.add_option("--max_len_a", dest="max_len_a", help="a for beam search (a*l+b)", type="float", default=1.3)
    parser.add_option("--max_len_b", dest="max_len_b", help="b for beam search (a*l+b)", type="int", default=5)
    parser.add_option("--len-penalty", dest="len_penalty_ratio", help="Length penalty", type="float", default=0.8)
    parser.add_option("--capacity", dest="total_capacity", help="Batch capacity", type="int", default=150)
    parser.add_option("--fp16", action="store_true", dest="fp16", default=False)
    return parser


def translate_batch(batch, generator, text_processor, verbose=False):
    src_inputs = batch["src_texts"].squeeze(0)
    src_mask = batch["src_pad_mask"].squeeze(0)
    tgt_inputs = batch["dst_texts"].squeeze(0)
    src_langs = batch["src_langs"].squeeze(0)
    dst_langs = batch["dst_langs"].squeeze(0)
    src_pad_idx = batch["pad_idx"].squeeze(0)
    src_text = None
    if verbose:
        src_ids = get_outputs_until_eos(text_processor.sep_token_id(), src_inputs, remove_first_token=True)
        src_text = list(map(lambda src: text_processor.tokenizer.decode(src.numpy()), src_ids))

    outputs = generator(src_inputs=src_inputs, src_sizes=src_pad_idx,
                        first_tokens=tgt_inputs[:, 0],
                        src_mask=src_mask, src_langs=src_langs, tgt_langs=dst_langs,
                        pad_idx=text_processor.pad_token_id())
    if torch.cuda.device_count() > 1:
        new_outputs = []
        for output in outputs:
            new_outputs += output
        outputs = new_outputs
    mt_output = list(map(lambda x: text_processor.tokenizer.decode(x[1:].numpy()), outputs))
    return mt_output, src_text


def build_data_loader(options, text_processor):
    print(datetime.datetime.now(), "Binarizing test data")
    assert options.src_lang is not None
    assert options.target_lang is not None
    src_lang = "<" + options.src_lang + ">"
    src_lang_id = text_processor.languages[src_lang]
    dst_lang = "<" + options.target_lang + ">"
    target_lang = text_processor.languages[dst_lang]
    fixed_output = [text_processor.token_id(dst_lang)]
    examples = []
    with open(options.input_path, "r") as s_fp:
        for i, src_line in enumerate(s_fp):
            if len(src_line.strip()) == 0: continue
            src_line = " ".join([src_lang, src_line, "</s>"])
            src_tok_line = text_processor.tokenize_one_sentence(src_line.strip().replace(" </s> ", " "))
            examples.append((src_tok_line, fixed_output, src_lang_id, target_lang))
            if i % 10000 == 0:
                print(i, end="\r")
    print("\n", datetime.datetime.now(), "Loaded %f examples", (len(examples)))
    test_data = dataset.MTDataset(examples=examples, max_batch_capacity=options.total_capacity, max_batch=options.batch,
                                  pad_idx=text_processor.pad_token_id(), max_seq_len=10000)
    pin_memory = torch.cuda.is_available()
    examples = None  # Make sure it gets collected
    return data_utils.DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=pin_memory)


def build_model(options):
    model = Seq2Seq.load(Seq2Seq, options.model_path, tok_dir=options.tokenizer_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    num_gpu = torch.cuda.device_count()
    generator = BeamDecoder(model, beam_width=options.beam_width, max_len_a=options.max_len_a,
                            max_len_b=options.max_len_b, len_penalty_ratio=options.len_penalty_ratio)
    if options.fp16:
        generator = amp.initialize(generator, opt_level="O2")
    if num_gpu > 1:
        generator = DataParallelModel(generator)
    return generator, model.text_processor


if __name__ == "__main__":
    parser = get_lm_option_parser()
    (options, args) = parser.parse_args()
    generator, text_processor = build_model(options)
    test_loader = build_data_loader(options, text_processor)
    sen_count = 0
    with open(options.output_path, "w") as writer:
        with torch.no_grad():
            for batch in test_loader:
                try:
                    mt_output, src_text = translate_batch(batch, generator, text_processor, options.verbose)
                    sen_count += len(mt_output)
                    print(datetime.datetime.now(), "Translated", sen_count, "sentences", end="\r")
                    if not options.verbose:
                        writer.write("\n".join(mt_output))
                    else:
                        writer.write("\n".join([y + "\n" + x + "\n****" for x, y in zip(mt_output, src_text)]))
                    writer.write("\n")
                except RuntimeError as err:
                    print("\n", repr(err))

    print(datetime.datetime.now(), "Translated", sen_count, "sentences")
    print(datetime.datetime.now(), "Done!")
