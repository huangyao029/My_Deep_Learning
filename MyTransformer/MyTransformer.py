# -*- coding:utf-8 -*-
# Author : Younger Huang
# Date : 2023.11.08

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import copy


def attention(query, key, value, mask = None, dropout = None):
    '''
    实现attention的公式:attention(Q,K,V)=softmax((QK^T/sqrt(d_k)))V
    '''
    # d_k：经过multi-head分割之后的维度大小，比如说如果没有multi-head，Q的维度是[T, D],
    # 那么，经过multi-head之后，每个head的维度就成了[T, d_k]
    d_k = query.size(-1)
    
    # 注意这里的torch.matmul的用法，这个属于[高维*高维]的dot product
    # [B, H, T, d_k] * [B, H, d_k, T] => [B, H, T, T]
    # B:batch size, H:head的数量， T:token的数量
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 整个transformer中，有两处地方用到了mask，这里是第一处，这里mask的作用在于：
    # 由于输入的长度很多时候都是不同的，所以需要在没有词的位置paddding上一个值，
    # 至于为什么这里padding的值是一个很小的负数，而不是0？那是因为这里还没有过softmax
    # 如果是0，那么过softmax之后的值可能就不一定是0了，但如果是一个很小的负数,过softmax
    # 后基本上就为0。
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        
    # 剩下的就是按照公式去写了。
    attn_weights = torch.softmax(scores, dim = -1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
        
    # [B, H, T, T] * [B, H, T, d_k] => [B, H, T, d_k]
    attn_output = torch.matmul(attn_weights, value)
    
    return attn_output, attn_weights 


class MultiHeadAttention(nn.Module):
    '''
    实现Multi-head Attention
    '''
    # 在实现分割成multi-head时，其实有两种思路，比如在生成Q时,可以定义多个线性层，比如我现在需要4个head
    # 那就定义4个Q的线性层。第二中就是transformer中的做法，先定义一个大的线性层，计算出Q之后，再分割成
    # 多个head的Q。那么其实这两种方法从本质上看没有区别，其实就是利用了卷积的可加性（矩阵的分配律）的性质。
    def __init__(self, d_model, num_heads, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
 
        # d_k表示multi-head中每个小head的维度
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        # 这里的输入输出都是d_model,但是其实输出也不一定要跟输入一样
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_weights = None
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask = None):
        # src_len为输入词的长度
        batch_size, src_len, _ = key.size()
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.view(batch_size, 1, 1, src_len)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
        
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)
        
        # 将QKVview成[B, H, T, d_k]
        # B:batch size, H:head的数量， T:token的数量
        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 套公式计算, 得到的x的shape为[B, H, T, d_k]
        x, self.attn_weights = attention(query, key, value, mask, self.dropout)
        
        # [B, H, T, d_k] => [B, T, H, d_k] => [B, T, H*d_k]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        
        # 最后再过一个线性层
        x = self.out_proj(x)
        
        return x
    
    
class PosititionwiseFeedForward(nn.Module):
    '''
    接在multi-head attention后面的Feed Forward层
    在原文章中就是这个公式：FFN(x)=max(0, xW_1+b_1)W_2+b_2
    '''
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(PosititionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
    

class LayerNorm(nn.Module):
    '''
    attention和feedforward层中都使用了layer normalization。
    '''
    def __init__(self, features, eps = 1e-6):
        super(LayerNorm, self).__init__()
        
        # nn.Parameter()可用于创建可训练的参数，这些参数会在模型训练的过程中自动更新，
        # nn.Parameter()继承自torch.Tensor(),因此它本质上也是一个Tensor，另外它还具有
        # 额外的属性requires_grad，用于指定参数是否需要计算梯度
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps
        
    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        norm = self.weight * (x - mean) / (std + self.eps) + self.bias
        return norm
    

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.coeff = math.sqrt(d_model)
        
    def forward(self, x):
        return self.embedding(x) * self.coeff
    

class PositionalEncoding(nn.Module):
    '''
    实现Positional Encoding的公式
    PE(pos, 2i) = sin(pos/10000^(2i/d_model))
    PE(pos, 2i+1) = sin(pos/10000^(2i/d_model))
    '''
    def __init__(self, d_model, dropout, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        
        # 通过公式简化，可以将公式变成：
        # PE(pos, 2i) = sin(pos * exp(2i*(-log10000/d_model)))
        # PE(pos, 2i+1) = cos(pos * exp(2i*(-log10000/d_model)))
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000) / d_model)
        )
        
        # 这是python中list的特殊用法之一
        # list切片的语法格式为list[start:end:step]，例如：
        # list1 = [1,2,3,4,5,6]
        # list1[0:5:1] -> [1,2,3,4,5]
        # list1[0:3:2] -> [1,3]
        # list1[0::2] -> [1,3,5]
        # list1[::-1] -> [6,5,4,3,2,1],即切片的变化量为-1，实现反转序列的功能
        # list1[1::-1] -> [2,1],即变化量为-1，从index=1开始，第一个输出是2，然后第二个是1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 在pytorch中，self.register_buffer()可以将tensor注册到模型的buffer()属性中，
        # 说人话就是，这个tensor能被模型的state_dict记录下来，同时还具备一个属性，就是
        # 这个tensor是一个持久态，不会有梯度传播给他，可以理解为模型的常数。那这个时候
        # 在一些场景就能用到，比如说，对比类似于torch.ones()这种常数，并不需要每次都去
        # 初始化或者计算这个常数，而是可以直接从buffer中取出来直接用。另外就是保存模型
        # 的时候就可以一起保存了。
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # 这里表示x=x+...就表示embedding的结果喝这里的position embedding结果相加,
        # 注意是add，而不是concate
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        
        return self.dropout(x)
    
    
class EncoderLayer(nn.Module):
    '''
    Encoder Layer Block
    '''
    # d_model:生成QKV时线性层的输出维度（在本代码中输出和输入都是d_model），
    # num_heads:head的数目
    # d_ff：在feed_forward层中，第一个线性层输出的维度
    # norm_first：是否先做LayerNorm
    def __init__(self, d_model, num_heads, d_ff, dropout, norm_first = True):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PosititionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_first = norm_first
    
    # norm在残差链接之前叫做pre-norm，norm在残差连接之后叫做post-norm。pre-norm的残差连接
    # 更明显，而post-norm的正则化效果更好
    def forward(self, x, mask):
        if self.norm_first:
            # x+self._sa_block()就是残差链接了，注意这里不是concate，而是add
            x = x + self._sa_block(self.norm1(x), mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, mask))
            x = self.norm2(x + self._ff_block(x))
        return x
    
    def _sa_block(self, x, mask):
        # 这里输入的QKV都是x，也就是说每个multiheadattention的block都是输入一个[T,D]的
        # 矩阵，输出也是这个size的矩阵
        x = self.self_attn(x, x, x, mask)
        
        return self.dropout1(x)
    
    def _ff_block(self, x):
        x = self.feed_forward(x)
        return self.dropout2(x)
    
    
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    '''
    Encoder Block
    '''
    def __init__(self, layer, num_layers, norm):
        super(Encoder, self).__init__()
        self.layers = clones(layer, num_layers)
        self.norm = norm
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x ,mask)
        
        if self.norm is not None:
            x = self.norm(x)
            
        return x
        
        
class DecoderLayer(nn.Module):
    '''
    Decoder Layer Block
    '''
    # 与EncoderLayer不同的是，DecoderLayer有三块，多了中间的编码-解码自注意力层，
    # 这个层接受来自编码器输出的结果作为这个块的KV，接受来自上一个self-attention的
    # 结果作为Q，计算之后再输出给feed-forward层
    def __init__(self, d_model, num_heads, d_ff, dropout, norm_first = True):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PosititionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm_first = norm_first
        
    # 在前面讲到，整个模型有两处mask，这里的src_mask是前面说的第一种mask，而tgt_mask
    # 就是第二种mask，在这里主要介绍一下tgt_mask的作用，这是为了在训练的时候仿真实际的
    # 情况，把未来看不到的信息给mask（遮盖）掉，mask的手段也跟第一种mask类似。
    def forward(self, x, memory, tgt_mask, src_mask):
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask)
            x = x + self._mha_block(self.norm2(x), memory, src_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask))
            x = self.norm2(x + self._mha_block(x, memory, src_mask))
            x = self.norm3(x + self._ff_block(x))
        return x
    
    # 注意这里的mask是第二种mask，而EncoderLayer中的mask是第一种mask
    def _sa_block(self, x, mask):
        x = self.self_attn(x, x, x, mask)
        return self.dropout1(x)
    
    # 相较于EncoderLayer中，多了这一个编码-解码自注意力层
    # 另外，这里的mask是第一种mask
    def _mha_block(self, x, memory, mask):
        x = self.cross_attn(x, memory, memory, mask)
        return self.dropout2(x)
    
    def _ff_block(self, x):
        x = self.feed_forward(x)
        return self.dropout3(x)
    
    
class Decoder(nn.Module):
    '''
    Decoder Block
    '''
    def __init__(self, layer, num_layers, norm):
        super(Decoder, self).__init__()
        self.layers = clones(layer, num_layers)
        self.norm = norm
        
    def forward(self, x, memory, tgt_mask, src_mask):
        
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        
        if self.norm is not None:
            x = self.norm(x)
            
        return x
    
    
class Generator(nn.Module):
    '''
    Generator主要是讲embedding转换为目标单词ID，即Transformer中Decoder嘴上ian的Linear+Softmax
    '''
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.proj(x)
        x = F.log_softmax(x, dim = -1)
        return x
    
    
class Transformer(nn.Module):
    '''
    Transformer!
    '''
    def __init__(
        self,
        d_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        d_ff,
        src_vocab_size,
        tgt_vocab_size,
        dropout = 0.1,
        norm_first = True,
    ):
        '''
        d_model:每个token的维度
        num_heads:heads的数目
        num_encoder_layers:encoder中encoder layers的数目
        num_decoder_layers:decoder中decoder layers的数目
        d_ff:feed forward中第一个linear的输出维度
        src_vocab_size:embedding相关
        tgt_vocab_size:embedding相关
        '''
        super(Transformer, self).__init__()
        
        # 对初始的句子进行编码，包括embedding喝position embedding，两者是相加，即add，不是concate
        self.src_embedding = nn.Sequential(
            Embeddings(src_vocab_size, d_model),
            PositionalEncoding(d_model, dropout),
        )
        self.tgt_embedding = nn.Sequential(
            Embeddings(tgt_vocab_size, d_model),
            PositionalEncoding(d_model, dropout),
        )
        
        # 一个encoder的layer，包括两部分，multi-head attention和FF层
        encoder_layer = EncoderLayer(
            d_model, num_heads, d_ff, dropout, norm_first,
        )
        
        # encoder完了之后做的LayerNorm
        encoder_norm = LayerNorm(d_model)
        self.encoder = Encoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        decoder_layer = DecoderLayer(
            d_model, num_heads, d_ff, dropout, norm_first,
        )
        decoder_norm = LayerNorm(d_model)
        self.decoder = Decoder(decoder_layer, num_decoder_layers, decoder_norm)
        
        self.generator = Generator(d_model, tgt_vocab_size)
        
        self.config = [d_model, num_heads, num_encoder_layers, num_decoder_layers,
                       d_ff, src_vocab_size, tgt_vocab_size, dropout, norm_first]
        
        self._init_parameters()
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, tgt_mask, src_mask)
        return output
    
    def encode(self, src, src_mask):
        src_embed = self.src_embedding(src)
        memory = self.encoder(src_embed, src_mask)
        return memory
        
    def decode(self, tgt, memory, tgt_mask, src_mask):
        tgt_embed = self.tgt_embedding(tgt)
        output = self.decoder(tgt_embed, memory, tgt_mask, src_mask)
        return output
    
    # 对模型参数进行初始化
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    # 保存模型
    def save(self, path):
        params = {
            'config' : self.config,
            'state_dict' : self.state_dict(),
        }
        torch.save(params, path)
    
    # @staticmethod可以用于将一个方法（函数）声明为静态方法，静态方法不会将类实例作为第一个参数传递
    # 而是直接访问类的命名空间，因此，静态方法可以在不创建类示例的情况下调用
    @staticmethod
    def load(model_path):
        params = torch.load(model_path, map_location = 'cpu')
        model = Transformer(*params['config'])
        model.load_state_dict(params['state_dict'])
        return model
    
    
if __name__ == '__main__':
    
    import torchinfo
    
    device = torch.device('cpu')
    model = Transformer(
        d_model = 512,
        num_heads = 8,
        num_encoder_layers = 6,
        num_decoder_layers = 6,
        d_ff = 2048,
        # src_vocab_size 和 tgt_vocab_size不一定准确，暂时随便用了个数
        src_vocab_size = 1024,
        tgt_vocab_size = 1024,
        dropout = 0.1
    )
    model.to(device)
    max_token_num = 1024
    print(torchinfo.summary(
        model,
        input_size = [(1, max_token_num), (1, max_token_num), (1, 8, max_token_num, max_token_num), (1, 8, max_token_num, max_token_num)],
        device = device,
        dtypes = [torch.IntTensor, torch.IntTensor, torch.FloatTensor, torch.FloatTensor],
        depth = 6
    ))