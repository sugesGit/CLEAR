import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math

class TransformerSelfattentionEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerSelfattentionEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.normalize = False
        if self.normalize:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(5)])

    def forward(self, x_in_list):
        x_list = x_in_list
        for layer in self.layers:
            x_list= layer(x_list)
        if self.normalize:
            x_list=[l(x)  for l, x in zip(self.layer_norm, x_list)]
        return x_list


class TransformerSelfattentionEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pre_self_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(5)])

        self.self_attns = nn.ModuleList([MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        ) for _ in range(5)])

        self.attn_mask = attn_mask

        self.res_dropout = res_dropout

    def forward(self, x_list):
        '''self attn'''
        residual = x_list

        x_list = [l(x) for l, x in zip(self.pre_self_attn_layer_norm, x_list)]
        output= [l(query=x, key=x, value=x) for l, x in zip(self.self_attns, x_list)]
        x_list=[ x for x, _ in output]

        x_list[0]=F.dropout(x_list[0], p=self.res_dropout , training=self.training)
        x_list[1]=F.dropout(x_list[1], p=self.res_dropout , training=self.training)
        x_list[2]=F.dropout(x_list[2], p=self.res_dropout , training=self.training)
        x_list[3]=F.dropout(x_list[3], p=self.res_dropout , training=self.training)
        x_list[4]=F.dropout(x_list[4], p=self.res_dropout , training=self.training)

        x_list = [r + x  for r, x in zip(residual, x_list)]
        
        return x_list
    
    
class TransformerRepresentationInteraction(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.res_dropout = res_dropout
        
        self.pre_encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.interrepresentations_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        
        self.pre_ffn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 =  nn.Linear(self.embed_dim, 4*self.embed_dim)
        self.fc2 = nn.Linear(4*self.embed_dim, self.embed_dim)

    def forward(self, x_list):
        '''inter-representations attn'''
        residual = x_list
        x_list = self.pre_encoder_attn_layer_norm(x_list)
        x_list = self.interrepresentations_attn(query=x_list, key=x_list, value=x_list)
        x_list, _ =  x_list
        x_list  = F.dropout(x_list, p=self.res_dropout, training=self.training)
        x_list = residual + x_list
        
        '''FFN'''
        residual = x_list
        x_list = self.pre_ffn_layer_norm(x_list)
        x_list = F.relu(self.fc1(x_list))
        x_list = F.dropout(x_list, p=self.res_dropout, training=self.training)
        
        x_list = F.relu(self.fc2(x_list))
        x_list = F.dropout(x_list, p=self.res_dropout, training=self.training)
        x_list = residual + x_list
            
        return x_list  

    
class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
    
class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super(Classifier, self).__init__()
        self.proj1 = nn.Linear(input_size, hidden_size)
        self.proj2 = nn.Linear(hidden_size, hidden_size)
        self.out_layer = nn.Linear(hidden_size, output_size)
        self.dropout = dropout

    def forward(self, input):
        x = F.dropout(F.relu(self.proj1(input)), p=self.dropout, training=self.training)
        x = F.dropout(F.relu(self.proj2(x)), p=self.dropout, training=self.training)
        output = self.out_layer(x).squeeze(-1)
        return output