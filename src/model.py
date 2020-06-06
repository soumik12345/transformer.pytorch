import torch


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
