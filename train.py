import torch
import spacy
from src.model import Transformer
from torchtext import data, datasets
from src.training import LabelSmoothing, DataIterator, batch_size_function


class MachineTranslationTrainer:

    def __init__(self, configs):
        self.configs = configs
        self.spacy_source = spacy.load(self.configs['source_language'])
        self.spacy_target = spacy.load(self.configs['target_language'])
        self.source = data.Field(
            tokenize=self.tokenize_source(),
            pad_token=self.configs['blank_word']
        )
        self.target = data.Field(
            tokenize=self.tokenize_target(),
            init_token=self.configs['begin_word'],
            eos_token=self.configs['end_word'],
            pad_token=self.configs['blank_word']
        )
        self.train, self.val, self.test = datasets.IWSLT.splits(
            exts=(
                self.configs['source_language_ext'],
                self.configs['target_language_ext']
            ),
            fields=(self.source, self.target),
            filter_pred=lambda x: len(vars(x)['src']) <= self.configs['max_length'] and
                                  len(vars(x)['trg']) <= self.configs['max_length']
        )
        self.source.build_vocab(
            self.train.src,
            min_freq=self.configs['min_freq']
        )
        self.target.build_vocab(
            self.train.trg,
            min_freq=self.configs['min_freq']
        )
        pad_index = self.target.vocab.stoi[self.configs['blank_word']]
        self.model = Transformer(
            len(self.source.vocab),
            len(self.target.vocab), n=6
        )
        self.model.cuda()
        self.criterion = LabelSmoothing(
            size=len(self.target.vocab),
            padding_idx=pad_index,
            smoothing=self.configs['smoothing']
        )
        self.criterion.cuda()
        self.train_iter = DataIterator(
            self.train, batch_size=self.configs['batch_size'],
            device=0, repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
            batch_size_fn=batch_size_function, train=True
        )
        self.valid_iter = DataIterator(
            self.val, batch_size=self.configs['batch_size'],
            device=0, repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
            batch_size_fn=batch_size_function, train=False
        )
        self.model_parameters = torch.nn.DataParallel(
            self.model, device_ids=self.configs['devices']
        )

    def tokenize_source(self, text):
        return [tok.text for tok in self.spacy_source.tokenizer(text)]

    def tokenize_target(self, text):
        return [tok.text for tok in self.spacy_target.tokenizer(text)]
