import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Transformerの基本的な構成要素であるMulti-Head AttentionとFeed Forward Networkを使って、
# シンプルな言語モデルを構築します。

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, ff_hidden_dim, max_length):
        super(TransformerLanguageModel, self).__init__()
        
        # Embedding層：トークンと位置情報をエンコードします
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=ff_hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 出力層
        self.fc_out = nn.Linear(embed_size, vocab_size)
        
        # 最大長の保存
        self.max_length = max_length

    def forward(self, x):
        # バッチサイズと系列長の取得
        batch_size, seq_length = x.size()
        
        # トークンの埋め込み
        token_embeddings = self.token_embedding(x)  # [batch_size, seq_length, embed_size]
        
        # 位置エンコーディング
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(batch_size, seq_length).to(x.device)
        position_embeddings = self.position_embedding(positions)  # [batch_size, seq_length, embed_size]
        
        # トークン埋め込みと位置埋め込みの加算
        embeddings = token_embeddings + position_embeddings  # [batch_size, seq_length, embed_size]
        
        # Transformer Encoderへの入力 (系列長とバッチサイズの順序に注意)
        embeddings = embeddings.permute(1, 0, 2)  # [seq_length, batch_size, embed_size]
        encoded_output = self.transformer_encoder(embeddings)  # [seq_length, batch_size, embed_size]
        
        # 出力層への入力 (再度順序を変換)
        encoded_output = encoded_output.permute(1, 0, 2)  # [batch_size, seq_length, embed_size]
        logits = self.fc_out(encoded_output)  # [batch_size, seq_length, vocab_size]
        
        return logits

# モデルの訓練に使用する関数
def train(model, data_loader, optimizer, criterion, num_epochs, device):
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 勾配の初期化
            optimizer.zero_grad()
            
            # モデルの出力を取得
            outputs = model(inputs)
            
            # 損失の計算（カテゴリクロスエントロピー）
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # 勾配の計算とパラメータの更新
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # エポックごとの損失を表示
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(data_loader):.4f}")

# モデルの定義に必要なパラメータ
vocab_size = 10000  # 語彙数
embed_size = 512  # 埋め込み次元数
num_heads = 8  # マルチヘッドアテンションのヘッド数
num_layers = 6  # エンコーダの層の数
ff_hidden_dim = 2048  # フィードフォワードネットワークの隠れ層の次元数
max_length = 100  # 最大系列長

# モデル、オプティマイザ、損失関数のインスタンス化
model = TransformerLanguageModel(vocab_size, embed_size, num_heads, num_layers, ff_hidden_dim, max_length)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルの訓練（データローダーは仮定として定義されているとします）
# train(model, data_loader, optimizer, criterion, num_epochs=10, device=device)

"""
関数とクラスの説明:
1. TransformerLanguageModel:
    - Transformerベースのシンプルな言語モデル。
    - Embedding層でトークンと位置情報を埋め込み、Transformerエンコーダで系列全体をエンコードします。
    - 出力層で各トークンの次に続くトークンの確率を予測します。

2. forward(x):
    - 入力シーケンスxを受け取り、トークンと位置の埋め込みを足し合わせた後、Transformerエンコーダに通します。
    - 最終的に語彙サイズに対応する出力を返します。

3. train:
    - モデルの訓練を行う関数。
    - データローダーからバッチごとにデータを取得し、損失を計算・勾配を更新します。
"""