import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model,
                 maxlen,
                 device):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        self.pos_embedding = pos_embedding.unsqueeze(0).to(device)

    def forward(self, x):
        x += self.pos_embedding[:, :x.size()[1], :]
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dff):
        super(FeedForwardNetwork, self).__init__()
        self.first_dense = nn.Linear(in_features=d_model, out_features=dff)
        self.second_dense = nn.Linear(in_features=dff, out_features=d_model)
        self.act_func = nn.ReLU()

    def forward(self, x):
        x = self.first_dense(x)
        x_act = self.act_func(x)
        x = self.second_dense(x_act)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.wq = nn.Linear(in_features=d_model, out_features=d_model)
        self.wk = nn.Linear(in_features=d_model, out_features=d_model)
        self.wv = nn.Linear(in_features=d_model, out_features=d_model)

        self.dense = nn.Linear(in_features=d_model, out_features=d_model)

    def split_heads(self, x, batch_size):
        x = torch.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return x.permute((0, 2, 1, 3))

    def forward(self, q, k, v, mask):
        batch_size = q.size()[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = scaled_attention.permute((0, 2, 1, 3))

        concat_attention = torch.reshape(
            scaled_attention,
            (batch_size, -1, self.d_model)
        )

        output = self.dense(concat_attention)

        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = torch.matmul(q, k.permute(0, 1, 3, 2))

        scaled_attention_logits = matmul_qk / math.sqrt(k.size()[-1])

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v)

        return output, attention_weights

