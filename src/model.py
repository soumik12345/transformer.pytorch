import torch
from copy import deepcopy
from .attention import MultiHeadedAttention
from .blocks import PositionWiseFeedForward, PositionalEncoding, Embeddings
from .encoder import Encoder
from .decoder import Decoder
from .encoder.blocks import EncoderLayer
from .decoder.blocks import DecoderLayer


class EncoderDecoder(torch.nn.Module):

    def __init__(self, encoder, decoder, source_embedding, target_embedding, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.generator = generator

    def encode(self, source, source_mask):
        embedding = self.source_embedding(source)
        encoder_output = self.encoder(embedding, source_mask)
        return encoder_output

    def decode(self, memory, source_mask, target, target_mask):
        embedding = self.target_embedding(target)
        decoder_output = self.decoder(embedding, memory, source_mask, target_mask)
        return decoder_output

    def forward(self, source, target, source_mask, target_mask):
        encoder_output = self.encode(source, source_mask)
        decoder_output = self.decode(encoder_output, source_mask, target, target_mask)
        return decoder_output


class Generator(torch.nn.Module):

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.linear = torch.nn.Linear(d_model, vocab)

    def forward(self, x):
        return torch.nn.functional.log_softmax(self.linear(x), dim=-1)


def Transformer(source_vocab, target_vocab, n=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    attention = MultiHeadedAttention(h, d_model)
    feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
    encoding = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(
            EncoderLayer(
                d_model, deepcopy(attention),
                deepcopy(feed_forward), dropout
            ), n
        ),
        Decoder(
            DecoderLayer(
                d_model, deepcopy(attention),
                deepcopy(attention), deepcopy(feed_forward), dropout
            ), n
        ),
        torch.nn.Sequential(
            Embeddings(d_model, source_vocab),
            deepcopy(encoding)
        ),
        torch.nn.Sequential(
            Embeddings(d_model, target_vocab),
            deepcopy(encoding)
        ),
        Generator(d_model, target_vocab)
    )
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform(p)
    return model
