# 各行の詳細な解説
import torch

PyTorchのコアライブラリをインポートします。
import torch.nn as nn

ニューラルネットワークのレイヤー定義などを含むtorch.nnモジュールをインポートします。
class MultiHeadAttention(nn.Module):

nn.Moduleを継承して、マルチヘッドアテンションのクラスを定義します。
def __init__(self, embed_size, num_heads):

コンストラクタメソッドで、埋め込み次元とヘッド数を受け取ります。
super(MultiHeadAttention, self).__init__()

親クラスnn.Moduleのコンストラクタを呼び出し、初期化を行います。
self.embed_size = embed_size

埋め込みサイズを保存します（埋め込みサイズは、入力ベクトルの次元数です）。
self.num_heads = num_heads

ヘッド数（マルチヘッドアテンションのヘッド数）を保存します。
self.head_dim = embed_size // num_heads

各ヘッドの次元数（head_dim）を計算します。embed_sizeをnum_headsで割ったものです。
assert embed_size % num_heads == 0

embed_sizeがnum_headsで割り切れることを確認します。割り切れないと、各ヘッドの次元が不整合になります。
self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)

Valueに対する線形変換を定義します。self.head_dimの次元を持つ値をself.head_dimに変換します。
self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)

Keyに対する線形変換を定義します。同様に、self.head_dimの次元を持つキーをself.head_dimに変換します。
self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

Queryに対する線形変換を定義します。self.head_dimの次元を持つクエリをself.head_dimに変換します。
self.fc_out = nn.Linear(num_heads * self.head_dim, embed_size)

最終出力の線形変換を定義します。全てのヘッドを結合した次元をembed_sizeに変換します。
def forward(self, query, key, value):

順伝播（フォワード）計算のメソッドを定義します。
N = query.shape[0]

バッチサイズ（N）を取得します。queryの最初の次元はバッチサイズです。
value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

Query、Key、Valueのそれぞれの長さを取得します。これらはシーケンスの長さに相当します。
queries = self.queries(query).view(N, query_len, self.num_heads, self.head_dim)

Queryに線形変換を適用し、その後viewでnum_heads個のヘッドに分割します。
keys = self.keys(key).view(N, key_len, self.num_heads, self.head_dim)

Keyに線形変換を適用し、同様にヘッドに分割します。
values = self.values(value).view(N, value_len, self.num_heads, self.head_dim)

Valueに線形変換を適用し、ヘッドに分割します。
energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)

QueryとKeyのドット積（内積）を計算し、エネルギー行列を生成します。torch.einsumはEinsteinの縮約記法で、効率的に行列積を計算します。
attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

エネルギー行列にスケーリングを適用し、ソフトマックスで正規化します。dim=3はアテンションスコアがキーの長さ方向で正規化されることを意味します。
out = torch.einsum("nhql,nlhd->nqhd", attention, values)

アテンションスコアをValueに掛けて、最終出力を計算します。
out = self.fc_out(out)

最終出力に対して線形変換を適用し、埋め込みサイズに戻します。
return out

最終的なアテンションの出力を返します。 