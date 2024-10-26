import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.amp import autocast, GradScaler
from torch.nn.utils.rnn import pad_sequence
import tokenizers
from DatasetsAndTorkenizer import loadTexts
import sys

#tokenizerインスタンスを作成&トレーニング
def loadTokenizer():
    #tokenizerをリロード
    tokenizer = tokenizers.Tokenizer.from_file("Neural Network/LLM/Transformer/tokenizer-wiki.json")
    return tokenizer

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, embed_size, num_heads, hidden_dim, num_layers, dropout=0.1):
        super(TransformerEncoderModel, self).__init__()
        
        # 埋め込み層: 入力のトークンIDを埋め込みベクトルに変換
        self.embedding = nn.Embedding(input_dim, embed_size)
        
        # Positional Encoding: シーケンスの位置情報を付与
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        
        # Transformer Encoder層(MHA->FFLのフローが全部この中で内部定義される)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_size, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=dropout
        )
        
        # エンコーダー全体の構築（複数のエンコーダーレイヤーを積み重ねる）
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 最終的な線形変換（デコードや分類用に出力を変換）
        self.fc_out = nn.Linear(embed_size, input_dim)

    def forward(self, src, src_mask=None):
        # 入力の埋め込み表現を計算
        src = self.embedding(src)  # (batch_size, seq_length, embed_size)
        
        # 位置情報を埋め込みに加算
        src = self.pos_encoder(src)
        
        # Transformer Encoderへの入力
        encoder_output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        
        # 出力を線形変換して最終結果を取得
        output = self.fc_out(encoder_output)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 位置エンコーディング行列を作成
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数インデックス
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数インデックス
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # モデルに登録
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 埋め込みベクトルに位置情報を加算
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# カスタムのcollate関数: パディングを行う
def collate_fn(batch):
    # バッチ内のテンソルを同じ長さにパディングする
    batch = [item for item in batch]
    return pad_sequence(batch, batch_first=True, padding_value=0)

# カスタムデータセットクラス
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode(text)
        input_ids = torch.tensor(encoding.ids)  # エンコード結果のトークンIDを取得しテンソルに変換
        return input_ids

if __name__ == "__main__":
    
    tokenizer = loadTokenizer()
    #ハイパーパラメータ
    input_dim = tokenizer.get_vocab_size() #トークンIDの総数
    embed_size = 256 #埋め込みベクトルの次元
    num_heads = 8 #マルチヘッドアテンションのヘッド数
    hidden_dim = 512 #FeedForward層の隠れ層次元数
    num_layers  = 6 #Transformerエンコーダーレイヤーの数
    dropout = 0.1  # ドロップアウト率
    num_epochs = 10  #エポック数

    #モデル定義
    model = TransformerEncoderModel(input_dim, embed_size, num_heads, hidden_dim, num_layers, dropout)

    # GPUが使用可能か確認
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # GPUが使用できない場合、メッセージを表示してプログラムを終了
    if device.type == 'cpu':
        print("Error: GPU is not available. Please ensure that a GPU is available and CUDA is properly installed.")
        sys.exit()  # プログラムを終了
    model.to(device)

    # サンプルデータを生成し、GPUに転送
    src = torch.randint(0, input_dim, (10, 32)).to(device)  # (シーケンス長, バッチサイズ)

    # マスクは任意（今回は使わない）
    src_mask = None

    # モデルの出力
    output = model(src, src_mask)

    print(output.shape)  # (シーケンス長, バッチサイズ, 語彙サイズ)

    # テキストデータをロード
    all_texts = loadTexts()
    # データセットのサイズを定義
    total_size = len(all_texts)

    # トレーニング、検証、テストの分割合計
    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.1

    # 分割サイズの計算
    train_size = int(train_ratio * total_size)
    validation_size = int(validation_ratio * total_size)
    test_size = total_size - train_size - validation_size  # 残りをテストに割り当て

    # トークナイザーを使ってテキストをトークン化
    tokenizer = loadTokenizer()

    # カスタムデータセットを作成
    full_dataset = TextDataset(all_texts, tokenizer)

    # データセットをランダムに分割
    train_dataset, validation_dataset, test_dataset = random_split(full_dataset, [train_size, validation_size, test_size])

    # DataLoaderを作成
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    validation_dataloader = DataLoader(validation_dataset, batch_size=8, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=collate_fn)

    # 最適化と損失関数の設定
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()

    scaler = GradScaler(device='cuda')
    # 学習ループ
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            # バッチデータをデバイスに転送
            inputs = batch.to(device)
            targets = inputs  # 次のトークンを予測するのでターゲットは入力と同じ
            
            # 勾配の初期化
            optimizer.zero_grad()
            
            # 自動混合精度での順伝播
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = loss_fn(outputs.view(-1, input_dim), targets.view(-1))

            # 逆伝播とパラメータ更新
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_dataloader)}")

    # 検証ループ（評価データセットがあれば使用）
    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        for batch in validation_dataloader:
            inputs = batch.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs.view(-1, input_dim), inputs.view(-1))
            total_val_loss += loss.item()

        print(f"Validation Loss: {total_val_loss/len(validation_dataloader)}")

    # モデルの保存
    torch.save(model.state_dict(), "transformer_encoder_model.pth")