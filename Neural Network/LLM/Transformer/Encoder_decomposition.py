from datasets import load_dataset
import numpy as np

def createTokenEmbeddingMatrix(tokenizer):
    #トークンIDの総数
    vocab_size = tokenizer.get_vocab_size()
    #埋め込みベクトルの次元(ハイパーパラメーター)
    embedding_dim = 768
    # 埋め込み行列をランダムに初期化 ({トークンIDの数}行{埋め込みベクトルの次元数}列の行列)
    embedding_matrix = np.random.randn(vocab_size, embedding_dim)

    return embedding_matrix

def extract_embeddings(tokenizer, embedding_matrix, input_text):
    # 1. トークナイズしてトークンIDを取得
    tokens = tokenizer.encode(input_text).ids
    
    # 2. トークンIDに対応するベクトルを抽出
    extracted_embeddings = embedding_matrix[tokens]
    
    return extracted_embeddings

def useTokenEmbeddingMatrix(embedding_matrix):
    # トークンIDリスト（例: [101, 7592, 2026]）
    input_tokens = [101, 7592, 2026]

    # トークンIDに対応するベクトルを取得
    embedded_tokens = [embedding_matrix[token_id] for token_id in input_tokens]

    # 結果の表示
    for i, token_id in enumerate(input_tokens):
        print(f"Token ID: {token_id}, Embedding Vector: {embedded_tokens[i][:5]}...")  # ベクトルの最初の5つの要素を表示

def createPositionalEmbeddingMatrix(tokenizer):
    #トークンIDの総数
    vocab_size = tokenizer.get_vocab_size()
    #埋め込みベクトルの次元(ハイパーパラメーター)
    embedding_dim = 768
    #位置符号行列を初期化
    positional_matrix = np.empty((vocab_size, embedding_dim))

    #位置符号行列をループして符号を追加
    #単語位置でループ(列の決定)
    for i in range(vocab_size):
        #埋め込み次元数でループ(行の決定)
        for k in range(embedding_dim//2): #切り捨て除算で位置を決定
            t = i / (10000 ** (2 * k / embedding_dim))
            positional_matrix[i, 2 * k] = np.sin(t)
            positional_matrix[i, 2 * k + 1] = np.cos(t)

    return positional_matrix

def combineTokenAndPositional(embedding_matrix, positional_matrix):
    embedding_matrix += positional_matrix
    return embedding_matrix

def softmax(x):
    ex = np.exp(x - np.max(x))  # オーバーフローを防ぐために最大値を引く
    return ex / (np.sum(ex, axis=-1, keepdims=True) + 1e-10)  #0除算防止に小さな値を加える

# 既存の埋め込み行列と重み行列を元にQ, K, Vを計算する関数を定義
def calculate_QKV(embedding_matrix, W_Q, W_K, W_V):
    """
    埋め込み行列に基づいてQuery, Key, Valueを計算する関数
    :param embedding_matrix: 埋め込み行列 (トークン数 × 埋め込み次元)
    :param W_Q: Queryの重み行列
    :param W_K: Keyの重み行列
    :param W_V: Valueの重み行列
    :return: Q, K, V行列
    """
    # Query行列を計算
    Q = np.dot(embedding_matrix, W_Q)  # トークン数 × 埋め込み次元
    # Key行列を計算
    K = np.dot(embedding_matrix, W_K)  # トークン数 × 埋め込み次元
    # Value行列を計算
    V = np.dot(embedding_matrix, W_V)  # トークン数 × 埋め込み次元
    return Q, K, V

def Single_Head_Attention(Q, K, V, embedding_dim):
    """
    Scaled Dot-Product Attentionの計算
    :param Q: Query行列(入力シーケンスのトークン数 × 埋め込みベクトルの次元数の形状を持つ行列)
    :param K: Key行列
    :param V: Value行列
    :return: Attentionによる出力
    """
    #埋め込みベクトルの次元(ハイパーパラメーター)
    embedding_dim = embedding_dim

    """
    ドット積 QKt: QueryとKeyの類似性を計算し、各トークン間の関連度を得る。
    QueryとKeyの転置行列のドット積を計算し、スケーリング
    Qのi行目は入力のi番目のトークン(1行768列の行列)⇒Kとのドット積で一つの値が出る
    scoresが持つのは入力長×入力長サイズを持つ行列となる
    """
    scores = np.dot(Q, K.T) / np.sqrt(embedding_dim)
    #scores = np.clip(scores, -500, 500)  # スコアを適度にクリッピング

    """
    ソフトマックスを適用してAttentionの重みを計算
    入力長×入力長サイズの確率値行列が得られる
    この確率値行列においてi行j列の値は、
    入力上のj番目のトークンが入力上のi番目のトークンに対してどれだけ注意を向けるかを示す
    """
    attention_weights = softmax(scores)
    #attention_weights = np.clip(attention_weights, 1e-5, 1) # 極端な小さな値を防ぐ
    """
    Attention重みとValue行列のドット積を計算して最終的な出力を得る
    Vは入力長×埋め込み次元数サイズの行列で各トークンの「意味」を持ち、
    Attention重みは入力長×入力長サイズの行列でそれぞれのトークンに対する注意量を持つ
    ドット積でVの1行とAttention重みの1列の重み付き和が得られる
    したがって、ドット積の結果1行×埋め込み次元数サイズの行列が各トークンについて得られる(入力トークン数×埋め込み次元数行列)
    """
    output = np.dot(attention_weights, V)

    return output, attention_weights

def Multi_Head_Attention(Q, K, V, embedding_dim, num_heads):
    """
    MHAとは埋め込み次元÷ヘッド数の次元をもって入力を多角的に解釈し、
    それを最後に結合することで元の入力と同じサイズの行列を出力することで、
    ヘッド数が8であれば8個分の解釈を持った埋め込み次元が作成できる
    """
    #埋め込みベクトルの次元(ハイパーパラメーター)
    embedding_dim = 768
    #MHAのヘッド数
    num_heads = num_heads
    """
    ヘッド数で埋め込み次元を割る理由：
    1．計算の効率化：各ヘッドごとの計算量を減らすことが可能
    2．分散した並列処理の実現：各ヘッドごとに異なる視点から入力を解釈することで、より精度の高い推論を実現
    """
    #各ヘッドの埋め込み次元
    d_k = embedding_dim // num_heads

    #各ヘッドで得られた重み行列と注意行列を保存するリスト
    heads_output = []
    heads_weights = []

    for i in range(num_heads):
        # Query, Key, Valueの重み行列（ランダムに初期化）
        W_Q = np.random.randn(embedding_dim, d_k) * np.sqrt(2 / embedding_dim)
        W_K = np.random.randn(embedding_dim, d_k) * np.sqrt(2 / embedding_dim)
        W_V = np.random.randn(embedding_dim, d_k) * np.sqrt(2 / embedding_dim)

        Q_head = np.dot(Q, W_Q)
        K_head = np.dot(K, W_K)
        V_head = np.dot(V, W_V)

        # 3. Single-Head Attentionを実行
        head_output, head_weights = Single_Head_Attention(Q_head, K_head, V_head, d_k)

        heads_output.append(head_output)
        heads_weights.append(head_weights)

    # 4. Concatenate heads
    concatenated_heads = np.concatenate(heads_output, axis=-1)

    # 5. 最終的な線形変換を実行（ヘッドの出力を統合）
    W_O = np.random.randn(concatenated_heads.shape[-1], embedding_dim)
    final_output = np.dot(concatenated_heads, W_O)

    return final_output, heads_weights

def relu(x):
    return np.maximum(0, x)

def forward(W1, W2, x, b1, b2):
    """
    一度ぶわっと広げた行列に入力の値をバラまいて計算して、
    ReLUで0以下を消し、得た結果の行列をもとの次元と同じ行列に再度バラまいて計算する
    結果一度解釈範囲を広げた後に元のサイズに圧縮された行列が得られる
    """
    # 第1の全結合層とReLU活性化関数
    x = np.dot(x, W1) + b1
    x = relu(x)
    
    # 第2の全結合層
    x = np.dot(x, W2) + b2
    return x

def feedforward(embedding_dim, x):
    """
    ここでは2層の順伝播NNとして定義する
    """
    #隠れ次元層(ハイパーパラメータ)
    hidden_dim = 2048    # 隠れ層の次元数

    W1 = np.random.randn(embedding_dim, hidden_dim) * np.sqrt(2 / embedding_dim)
    W2 = np.random.randn(hidden_dim, embedding_dim) * np.sqrt(2 / hidden_dim)
    b1 = np.zeros((1, hidden_dim))
    b2 = np.zeros((1, embedding_dim))

    output = forward(W1, W2, x, b1, b2)

    return output

def Layer_Normalization(x):
    """
    データの各行（トークン）の特徴量が、平均0、標準偏差1の正規化されたベクトルに変換されます.
    これにより、異なる層での分布の変化に依存せず、安定した学習が可能になります.
    """
    epsilon=1e-6
    """Layer Normalization"""
    # 各行に対する平均を計算 (axis=-1は行方向で計算)
    mean = np.mean(x, axis=-1, keepdims=True)
    
    # 各行に対する標準偏差を計算 (axis=-1は行方向で計算)
    std = np.std(x, axis=-1, keepdims=True)
    
    # 入力xを正規化 (mean=0, std=1になるようにスケーリング)
    return (x - mean) / (std + epsilon)

def residual_connection(x, sublayer_output):
    """
    残差結合: sublayer_output (MHA またはフィードフォワード層の出力) に元の入力 x を足し合わせる。
    さらに、足し合わせたものにLayer Normalizationを適用。
    """
    return Layer_Normalization(x + sublayer_output)

def dropout(x):
    """
    X: 入力データ
    drop_prob: ドロップアウト率（例: 0.5）
    """
    #ドロップアウト率(ハイパーパラメータ)
    drop_prob = 0.9

    """
    入力データのサイズをPython：アンパック機能で渡して一様乱数分布を作成
    drop_probの確率で値を0にする
    """
    dropout_mask = np.random.rand(*x.shape) > drop_prob

    #入力データにマスクを適用して一部の値を無効化
    x_dropped = x*dropout_mask

    #ドロップアウト率に応じてスケーリング
    #ドロップアウト率に応じてスケーリングすることで、全てのニューロンを使用する場合と期待値が一致するようにする
    x_scaled = x_dropped / (1.0 - drop_prob)

    return x_scaled

#正規版関数定義
def Tokenize(input_text):
    tokenizer = loadTokenizer()
    #埋め込み行列作成
    embed_matrix = createTokenEmbeddingMatrix(tokenizer)
    
    positional_matrix = createPositionalEmbeddingMatrix(tokenizer)
    embed_matrix = combineTokenAndPositional(embed_matrix, positional_matrix)
    extracted_embeddings = extract_embeddings(tokenizer, embed_matrix, input_text)
    return extracted_embeddings #(input_vocab_size, embedding_dim)

def MHA(extracted_embeddings, embedding_dim, num_heads):
    """
    1. トークン化済み入力行列からQ, V, Kを算出
    2. MHAでいくつかに分割して計算、結合して出力と注意重み行列を獲得
    3. MHA出力に対して残差結合
    4. 残差結合出力に対してドロップアウト
    """
    # Query, Key, Valueの重み行列（ランダムに初期化）
    W_Q = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(2 / embedding_dim)
    W_K = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(2 / embedding_dim)
    W_V = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(2 / embedding_dim)
    Q, K, V = calculate_QKV(extracted_embeddings, W_Q, W_K, W_V) #(input_vocab_size, embedding_dim)
    final_output, heads_weights = Multi_Head_Attention(Q, K, V, embedding_dim, num_heads) #final_output:(input_vocab_size, embedding_dim)
    output_with_dropout = dropout(final_output) #(input_vocab_size, embedding_dim)
    residual_output = residual_connection(extracted_embeddings, output_with_dropout) #(input_vocab_size, embedding_dim)

    return residual_output, heads_weights

def FFL(ex_output):
    output = feedforward(embedding_dim, ex_output)
    output_with_dropout = dropout(output) #(input_vocab_size, embedding_dim)
    residual_output = residual_connection(ex_output, output_with_dropout) #(input_vocab_size, embedding_dim)

    return residual_output

def Transformer_EncoderOnly(input_text):
    embedding_dim = 768
    num_heads = 8
    
    # トークナイズして埋め込みを取得
    extracted_embeddings = Tokenize(input_text)
    
    # MHA と FFL を1度に処理
    output, heads_weights = MHA(extracted_embeddings, embedding_dim, num_heads)
    output = FFL(output)
    
    return output


if __name__ == "__main__":
    # トークン数と埋め込み次元数の例
    E_len = 100  # 入力シーケンス長
    embedding_dim = 384  # 埋め込みベクトルの次元

    # データの例を表示
    #for i in range(10):
        #print(train_dataset[i])
        #print(valid_dataset[i])
        #print(test_dataset[i])

    #tokenizerインスタンスを作成&トレーニング
    #tokenizer()
    tokenizer = loadTokenizer()
    #useTokenizer(tokenizer)
    embedding_matrix = createTokenEmbeddingMatrix(tokenizer)
    #print(embedding_matrix)
    positional_matrix = createPositionalEmbeddingMatrix(tokenizer)
    #print(positional_matrix)
    embedding_matrix = combineTokenAndPositional(embedding_matrix, positional_matrix)
    #print(embedding_matrix)
    #useTokenEmbeddingMatrix(embedding_matrix)

    # Query, Key, Valueの重み行列（ランダムに初期化）
    #W_Q = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(2 / embedding_dim)
    #W_K = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(2 / embedding_dim)
    #W_V = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(2 / embedding_dim)
    # Q, K, Vの計算
    #Q, K, V = calculate_QKV(embedding_matrix, W_Q, W_K, W_V)
    #output, attention_weights = Multi_Head_Attention(Q, K, V, embedding_dim, 8)
    #print(output)
    #print(attention_weights)

    # ダミー入力データ (1つのシーケンスのトークンに対する埋め込みベクトル)
    x = np.random.randn(1, embedding_dim)  # 1つのトークンの埋め込みベクトル
    output = feedforward(embedding_dim, x) #本来はMHAのoutputをxに渡す
    print(output)

    #残差結合と層正規化
    output_from_mha = np.random.randn(1, embedding_dim)  # MHAやフィードフォワードからの出力
    x_input = np.random.randn(1, embedding_dim)  # 元の入力

    # 残差結合
    output_with_residual = residual_connection(x_input, output_from_mha)

    print(output_with_residual)

    #Dropout
    # テスト用のデータ
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # ドロップアウトを適用
    X_after_dropout = dropout(X)

    print("ドロップアウト後の入力データ:")
    print(X_after_dropout)

    #正規版
