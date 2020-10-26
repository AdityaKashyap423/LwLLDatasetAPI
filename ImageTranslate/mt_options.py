class TrainOptions:
    def __init__(self):
        self.train_path = None
        self.dev_path = None
        self.tokenizer_path = None
        self.cache_size = 300
        self.model_path = None
        self.pretrained_path = None
        self.num_epochs = 100
        self.clip = 1
        self.batch = 6000
        self.mask_prob = 0.5
        self.learning_rate = 5e-5
        self.warmup = 4000
        self.step = 100000
        self.max_grad_norm = 1.0
        self.continue_train = False
        self.dropout = 0.2
        self.encoder_layer = 6
        self.embed_dim = 768
        self.intermediate_layer_dim = 3072
        self.total_capacity = 600
        self.lm_path = None
        self.dict_path = None
        self.beam_width = 5
        self.max_len_a = 1.3
        self.max_len_b = 5
        self.len_penalty_ratio = 0.8
        self.max_seq_len = 175
        self.lang_decoder = True
        self.nll = False
        self.fp16 = True
        self.mt_train_path = None
        self.mt_dev_path = None
        self.finetune_step = 0
        self.mass_train_path = None
        self.image_dir = None
        self.img_capacity = 50
        self.max_image = 32
        self.resnet_depth = 1
        self.bt_langs = "ar,en"
        self.mm_mode = "mixed"
        self.decoder_layer = 6
        self.ignore_mt_mass = True
        self.tie_embed = False


class TranslateOptions:
    def __init__(self):
        self.input_path = None
        self.src_lang = "ar"
        self.target_lang = "en"
        self.output_path = None
        self.batch = 512
        self.tokenizer_path=None
        self.model_path = None
        self.verbose = False
        self.beam_width = 5
        self.max_len_a = 1.3
        self.max_len_b = 5
        self.len_penalty_ratio = 0.8
        self.total_capacity = 150
        self.fp16 = True
