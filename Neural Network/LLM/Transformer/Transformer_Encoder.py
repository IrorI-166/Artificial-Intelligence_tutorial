import torch
import torch.nn as nn
import tokenizers

#tokenizerインスタンスを作成&トレーニング
def tokenizer():
    #tokenizerのインスタンスを作成
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="[UNK]"))
    #trainerのインスタンスを初期化
    trainer = tokenizers.trainers.BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    #tokenizerにpre_trainer属性を追加
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)
    
    def forward(self, query, key, value):
        N = query.shape[0]  # batch size
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]
        
        # Query, Key, Valueの線形変換と分割 (num_heads個に)
        queries = self.queries(query).view(N, query_len, self.num_heads, self.head_dim)
        keys = self.keys(key).view(N, key_len, self.num_heads, self.head_dim)
        values = self.values(value).view(N, value_len, self.num_heads, self.head_dim)
        
        # アテンションスコアの計算
        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)  # ドット積
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)  # スケーリング&ソフトマックス
        
        # Valueに対するアテンション
        out = torch.einsum("nhql,nlhd->nqhd", attention, values).reshape(N, query_len, self.num_heads * self.head_dim)
        
        # 最後に全体を線形変換して出力
        out = self.fc_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_size, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self, embed_size, heads, hidden_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ffn = FeedForward(embed_size, hidden_dim)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # マルチヘッドアテンションの適用
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # フィードフォワードネットワークの適用
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x

class Encoder(nn.Module):
    def __init__(self, num_layers, embed_size, heads, hidden_dim, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(embed_size, heads, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, hidden_dim, dropout, max_length):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_length, embed_size)
        self.encoder = Encoder(num_layers, embed_size, heads, hidden_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 入力の埋め込みと位置埋め込みを加算
        seq_length = x.shape[1]
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(x.size(0), seq_length)
        x = self.dropout(self.embedding(x) + self.pos_embedding(positions))

        # エンコーダーを通して出力を得る
        out = self.encoder(x, mask)
        return out

if __name__ == "__main__":