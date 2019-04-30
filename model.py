import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class LinearAutoEncoder(nn.Module):
    def __init__(self):
        super(LinearAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5000, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Linear(512, 5000),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MultiHeadAtten(nn.Module):
    """
    Multi head attetnion

    ![](https://raw.githubusercontent.com/LinXueyuanStdio/scRNN-seq/master/art/3.png)
    """

    def __init__(self, atten_unit, encode_size, num_heads=8, dropout=0.1):
        super(MultiHeadAtten, self).__init__()

        model_dim = encode_size * 2

        self.attention = atten_unit
        self.num_heads = num_heads
        self.dim_per_head = d_k = d_v = model_dim // num_heads

        self.linear_q = nn.Linear(model_dim, num_heads * d_k)
        self.linear_k = nn.Linear(model_dim, num_heads * d_k)
        self.linear_v = nn.Linear(model_dim, num_heads * d_v)

        unit_encode_size = d_k // 2
        self.attention = atten_unit(unit_encode_size, dropout)
        self.linear_final = nn.Linear(num_heads * d_v, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, query, key, value, atten_mask=None):
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = query.size(0)

        # 线性连接层
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)

        # split by heads
        query = query.view(batch_size * num_heads, -1, dim_per_head)
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)

        if atten_mask:
            atten_mask = atten_mask.repeat(num_heads, 1, 1)

        # scaled dot product attention
        context, atten = self.attention(query, key, value, atten_mask)

        # 合并
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # 线性连接层
        output = self.linear_final(context)

        output = self.dropout(output)

        # 残差连接 + norm
        output = self.layer_norm(residual + output)

        return output, atten


class ScaledDotProductAtten(nn.Module):
    """
    Scaled dot-product attention mechainsm

    公式：
        $  Attention(Q, K, V) = softmax(\frac{Q K^T}{\sqrt{d_k}})*V $

    ![](https://raw.githubusercontent.com/LinXueyuanStdio/scRNN-seq/master/art/2.png)
    """

    def __init__(self, encode_size, atten_dropout=0.1):
        super(ScaledDotProductAtten, self).__init__()

        encode_size = 2 * encode_size
        self.scale = encode_size ** -0.5
        self.dropout = nn.Dropout(atten_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value, atten_mask=None):
        '''
        前向传播.

        Args:
                Q: Queries，[B, L_q, D_q]
                K: Keys，[B, L_k, D_k]
                V: Values，[B, L_v, D_v]，一般来说就是k
                scale: 缩放因子，一个浮点标量
                attn_mask: Masking，[B, L_q, L_k]
        Returns:
                上下文张量和attetention张量
        '''
        atten = torch.bmm(query, key.transpose(1, 2)) * self.scale
        if atten_mask:
            atten.masked_fill_(atten_mask, -np.inf)
        atten = self.softmax(atten)
        atten = self.dropout(atten)
        context = torch.bmm(atten, value)
        return context, atten


class ConcatAtten(nn.Module):
    """Additive Attention"""

    def __init__(self, encode_size, atten_dropout=0.0):
        super(ConcatAtten, self).__init__()

        self.Wc1 = nn.Linear(2 * encode_size, encode_size, bias=False)
        self.Wc2 = nn.Linear(2 * encode_size, encode_size, bias=False)
        self.vc = nn.Linear(encode_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(atten_dropout)

    def forward(self, query, key, value, atten_mask=None):
        q = self.Wc1(query).unsqueeze(1)
        k = self.Wc2(key).unsqueeze(2)
        sjt = self.vc(torch.tanh(q + k)).squeeze()

        if atten_mask:
            sjt.masked_fill_(atten_mask, -np.inf)

        atten = self.softmax(sjt)
        atten = self.dropout(atten)
        context = torch.bmm(atten, value)
        return context, atten


class BilinearAtten(nn.Module):

    def __init__(self, encode_size: int, atten_dropout=0.0):
        super(BilinearAtten, self).__init__()

        self.Wb = nn.Linear(2 * encode_size, encode_size, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(atten_dropout)

    def forward(self, query, key, value, atten_mask=None):
        s1 = self.Wb(query).transpose(2, 1)
        sjt = torch.bmm(key, s1)
        if atten_mask:
            sjt.masked_fill_(atten_mask, -np.inf)
        atten = self.softmax(sjt)
        atten = self.dropout(atten)
        context = torch.bmm(value, atten)
        return context, atten


class DotAtten(nn.Module):
    """Dot Product Attention"""

    def __init__(self, encode_size: int, atten_dropout=0.0):
        super(DotAtten, self).__init__()

        self.Wd = nn.Linear(2 * encode_size, encode_size, bias=False)
        self.vd = nn.Linear(encode_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(atten_dropout)

    def forward(self, query, key, value, atten_mask=None):
        q = query.unsqueeze(1)
        k = query.unsqueeze(2)
        sjt = self.vd(torch.tanh(self.Wd(q * k))).squeeze()

        if atten_mask:
            sjt.masked_fill_(atten_mask, -np.inf)

        atten = self.softmax(sjt)
        atten = self.dropout(atten)
        context = torch.bmm(atten, value)
        return context, atten


class MinusAtten(nn.Module):
    """MinusAttention"""

    def __init__(self, encode_size, atten_dropout=0.1):
        super(MinusAtten, self).__init__()

        self.Wm = nn.Linear(2 * encode_size, encode_size, bias=False)
        self.vm = nn.Linear(encode_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(atten_dropout)

    def forward(self, query, key, value, atten_mask=None):

        q = query.unsqueeze(1)
        k = query.unsqueeze(2)
        sjt = self.vm(torch.tanh(self.Wm(q - k))).squeeze()

        if atten_mask:
            sjt.masked_fill_(atten_mask, -np.inf)

        atten = self.softmax(sjt)
        atten = self.dropout(atten)
        context = torch.bmm(atten, value)
        return context, atten


class SelfAtten(nn.Module):
    """SelfAttention"""

    def __init__(self, model_dim, atten_dropout=0.1):
        super(SelfAtten, self).__init__()
        self.Ws = nn.Linear(2 * model_dim, model_dim, bias=False)
        self.vs = nn.Linear(model_dim, 1, bias=False)
        self.dropout = nn.Dropout(atten_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, hidden, atten_mask=None):
        query = hidden.unsqueeze(1)
        key = hidden.unsqueeze(2)
        value = hidden
        sjt = self.vs(torch.tanh(self.Ws(query * key))).squeeze()
        if atten_mask:
            sjt.masked_fill_(atten_mask, -np.inf)
        atten = self.softmax(sjt)
        atten = self.dropout(atten)
        context = torch.bmm(atten, value)
        return context, atten


class PositionwiseFeedForward(nn.Module):
    """ a　two layer feed forward"""

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(residual + output)
        return output


class TransformerEncoder(nn.Module):
    """
    Transformer encoder

    ![](https://raw.githubusercontent.com/LinXueyuanStdio/scRNN-seq/master/art/0.png)
    """

    def __init__(self, encode_size=512, num_heads=8, ffn_dim=2018, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.attention = MultiHeadAtten(ScaledDotProductAtten(encode_size, dropout),
                                        encode_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(encode_size, ffn_dim, dropout)

    def forward(self, inputs, atten_mask=None):
        context, atten = self.attention(inputs, inputs, inputs, atten_mask)
        output = self.feed_forward(context)
        return output, atten


class LinearAttnAutoEncoder(nn.Module):
    def __init__(self):
        super(LinearAttnAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5000, 512),
            nn.ReLU(True),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Linear(512, 5000),
            nn.Sigmoid())
        self.attention = ScaledDotProductAtten(5000)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x, attn = self.attention(x)
        return x
