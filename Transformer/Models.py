import torch.nn as nn
from Transformer.Layers import EncoderLayer
from Transformer.Modules import PositionalEncoding
from Transformer.utils import create_padding_mask
import math
import torch


class Encoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 input_vocab_size,
                 maximum_position_encoding,
                 dropout_rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, maximum_position_encoding)

        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):

        x = self.embedding(x)
        x *= math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        return x


class Transformer(nn.Module):
    def __init__(self,
                 input_vocab_size,
                 target_vocab_size,
                 num_layers=6,
                 d_model=512,
                 num_heads=8,
                 dff=2048,
                 position_encoding_input=512,
                 position_encoding_target=512,
                 dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            position_encoding_input
        )

        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self,
                inp,
                tar,
                enc_padding_mask=None,
                look_ahead_mask=None,
                dec_padding_mask=None):
        enc_output = self.encoder(inp, enc_padding_mask)

        final_output = self.final_layer(enc_output)

        return final_output


if __name__ == '__main__':

    x = torch.randint(0, 5000, (4, 100))
    y = torch.randint(0, 5000, (4, 100))

    padding_mask = create_padding_mask(x)

    model = Transformer(
        input_vocab_size=5000,
        target_vocab_size=5000
    )

    outputs = model(
        inp=x,
        tar=y,
        enc_padding_mask=padding_mask
    )

    print(outputs)
