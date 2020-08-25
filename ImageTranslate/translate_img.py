import datetime
from optparse import OptionParser

import torch
import torch.utils.data as data_utils
from apex import amp
from torch.nn.utils.rnn import pad_sequence

import dataset
from image_model import ImageCaptioning, Caption2Image
from parallel import DataParallelModel
from seq2seq import Seq2Seq
from seq_gen import BeamDecoder, get_outputs_until_eos


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
    parser.add_option("--caption-model", dest="caption_model_path", metavar="FILE", default=None)
    parser.add_option("--verbose", action="store_true", dest="verbose", help="Include input!", default=False)
    parser.add_option("--beam", dest="beam_width", type="int", default=4)
    parser.add_option("--max_len_a", dest="max_len_a", help="a for beam search (a*l+b)", type="float", default=1.3)
    parser.add_option("--max_len_b", dest="max_len_b", help="b for beam search (a*l+b)", type="int", default=5)
    parser.add_option("--len-penalty", dest="len_penalty_ratio", help="Length penalty", type="float", default=0.8)
    parser.add_option("--capacity", dest="total_capacity", help="Batch capacity", type="int", default=150)
    parser.add_option("--fp16", action="store_true", dest="fp16", default=False)
    return parser


def translate_batch(batch, txt2img, generator, text_processor, verbose=False):
    pad_idx = text_processor.pad_token_id()
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

    gen_module = generator.module if hasattr(generator, "module") else generator
    max_len = min(int(gen_module.max_len_a * src_inputs.size(1) + gen_module.max_len_b), 512)

    image_embed = txt2img(src_inputs, src_mask, src_langs)
    image_embed = image_embed.view(image_embed.size(0), 49, -1)
    outputs = generator(first_tokens=tgt_inputs[:, 0], max_len=max_len,
                        tgt_langs=dst_langs, image_embed=image_embed,
                        pad_idx=pad_idx)
    if torch.cuda.device_count() > 1:
        new_outputs = []
        for output in outputs:
            new_outputs += output
        outputs = new_outputs
    mt_output = list(map(lambda x: text_processor.tokenizer.decode(x[1:].numpy()), outputs))

    output_padded = pad_sequence(outputs, batch_first=True, padding_value=pad_idx)
    output_pad_idx = (output_padded != pad_idx)
    output_image_embed = txt2img(output_padded, output_pad_idx, dst_langs)
    output_image_embed = output_image_embed.view(output_image_embed.size(0), 49, -1)

    second_outputs = generator(first_tokens=src_inputs[:, 0], max_len=max_len,
                               tgt_langs=src_langs, image_embed=output_image_embed,
                               pad_idx=pad_idx)
    if torch.cuda.device_count() > 1:
        new_outputs = []
        for output in second_outputs:
            new_outputs += output
        second_outputs = new_outputs
    mt_2nd_output = list(map(lambda x: text_processor.tokenizer.decode(x[1:].numpy()), second_outputs))

    output_2nd_padded = pad_sequence(second_outputs, batch_first=True, padding_value=pad_idx)
    output_2nd_pad_idx = (output_2nd_padded != pad_idx)
    output_3rd_image_embed = txt2img(output_2nd_padded, output_2nd_pad_idx, src_langs)
    output_3rd_image_embed = output_3rd_image_embed.view(output_3rd_image_embed.size(0), 49, -1)

    third_outputs = generator(first_tokens=tgt_inputs[:, 0], max_len=max_len,
                              tgt_langs=dst_langs, image_embed=output_3rd_image_embed,
                              pad_idx=pad_idx)
    if torch.cuda.device_count() > 1:
        new_outputs = []
        for output in third_outputs:
            new_outputs += output
        third_outputs = new_outputs
    mt_3rd_output = list(map(lambda x: text_processor.tokenizer.decode(x[1:].numpy()), third_outputs))

    return mt_output, src_text, mt_2nd_output, mt_3rd_output


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
        for src_line in s_fp:
            if len(src_line.strip()) == 0: continue
            src_line = " ".join([src_lang, src_line, "</s>"])
            src_tok_line = text_processor.tokenize_one_sentence(src_line.strip().replace(" </s> ", " "))
            examples.append((src_tok_line, fixed_output, src_lang_id, target_lang))
    print(datetime.datetime.now(), "Loaded %f examples", (len(examples)))
    test_data = dataset.MTDataset(examples=examples,
                                  max_batch_capacity=options.total_capacity, max_batch=options.batch,
                                  pad_idx=text_processor.pad_token_id(), max_seq_len=10000)
    pin_memory = torch.cuda.is_available()
    return data_utils.DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=pin_memory)


def build_model(options):
    model = Caption2Image.load(options.model_path, options.tokenizer_path)
    caption_model = Seq2Seq.load(ImageCaptioning, options.caption_model_path, tok_dir=options.tokenizer_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    caption_model = caption_model.to(device)
    num_gpu = torch.cuda.device_count()
    generator = BeamDecoder(caption_model, beam_width=options.beam_width, max_len_a=options.max_len_a,
                            max_len_b=options.max_len_b, len_penalty_ratio=options.len_penalty_ratio)
    if options.fp16:
        model = amp.initialize(model, opt_level="O2")
        generator = amp.initialize(generator, opt_level="O2")
    if num_gpu > 1:
        model = DataParallelModel(model)
        generator = DataParallelModel(generator)
    return model, generator, model.text_processor


if __name__ == "__main__":
    parser = get_lm_option_parser()
    (options, args) = parser.parse_args()
    txt2img_model, generator, text_processor = build_model(options)
    test_loader = build_data_loader(options, text_processor)
    sen_count = 0
    with open(options.output_path, "w") as writer:
        with torch.no_grad():
            for batch in test_loader:
                mt_output, src_text, mt_2nd_output, mt_3rd_output = translate_batch(batch, txt2img_model, generator,
                                                                                    text_processor,
                                                                                    options.verbose)

                sen_count += len(mt_output)
                print(datetime.datetime.now(), "Translated", sen_count, "sentences", end="\r")
                if not options.verbose:
                    writer.write("\n".join(mt_output))
                else:
                    writer.write("\n".join(
                        [y + "\n" + x + "\n" + z + "\n" + f + "\n****" for x, y, z, f in
                         zip(mt_output, src_text, mt_2nd_output, mt_3rd_output)]))
                writer.write("\n")

    print(datetime.datetime.now(), "Translated", sen_count, "sentences")
    print(datetime.datetime.now(), "Done!")
