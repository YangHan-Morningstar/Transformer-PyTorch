import torch.nn as nn
from transformer.Layers import EncoderLayer, DecoderLayer
from transformer.Modules import PositionalEncoding
from transformer.utils import create_masks
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
                 dropout_rate,
                 device):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, maximum_position_encoding, device)

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


class Decoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 target_vocab_size,
                 maximum_position_encoding,
                 dropout_rate,
                 device):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, maximum_position_encoding, device)

        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attention_weights = {}

        x = self.embedding(x)
        x *= math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, att_block1, att_block2 = self.dec_layers[i](
                x, enc_output, look_ahead_mask, padding_mask
            )

            attention_weights['decoder_layer{}_block1'.format(i + 1)] = att_block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = att_block2

        return x, attention_weights


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
                 dropout_rate=0.1,
                 device="cpu"):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            input_vocab_size,
            position_encoding_input,
            dropout_rate,
            device
        )

        self.decoder = Decoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            target_vocab_size,
            position_encoding_target,
            dropout_rate,
            device
        )

        self.final_layer = nn.Linear(d_model, target_vocab_size)

    def forward(self,
                inp,
                tar,
                enc_padding_mask=None,
                look_ahead_mask=None,
                dec_padding_mask=None):
        enc_output = self.encoder(inp, enc_padding_mask)

        dec_output, attention_weights = self.decoder(
            tar, enc_output, look_ahead_mask, dec_padding_mask
        )

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights


if __name__ == '__main__':

    x = torch.tensor([
        [456, 123, 543, 3213, 56, 552, 2312, 323, 2456, 0, 0, 0, 0],
        [4, 1, 5, 33, 545, 312, 0, 0, 0, 0, 0, 0, 0]
    ])
    y = torch.tensor([
        [998, 893, 342, 313, 1212, 0, 0, 0],
        [2313, 3434, 321, 3231, 3123, 3123, 32, 0]
    ])

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(x, y)

    model = Transformer(
        input_vocab_size=5000,
        target_vocab_size=5000
    )

    outputs, _ = model(
        inp=x,
        tar=y,
        enc_padding_mask=enc_padding_mask,
        look_ahead_mask=combined_mask,
        dec_padding_mask=dec_padding_mask
    )

    print(model)

    print(outputs)
