from datasets import load_dataset
import tokenizers
import numpy as np
#from tokenizers import Tokenizer
#from tokenizers.trainers import BpeTrainer
#from tokenizers.pre_tokenizers import Whitespace
#from tokenizers.models import BPE
#from tokenizers.processors import TemplateProcessing

# データセットをロード
ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

# トレーニング/検証/テストデータセットを取得
train_dataset = ds['train']
valid_dataset = ds['validation']
test_dataset = ds['test']

#tokenizerインスタンスを作成&トレーニング
def tokenizer():
    #tokenizerのインスタンスを作成
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="[UNK]"))
    #trainerのインスタンスを初期化
    trainer = tokenizers.trainers.BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    #tokenizerにpre_trainer属性を追加
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    # データセットからテキストを抽出 (train, valid, test からトレーニング)
    all_texts = []
    for dataset in [train_dataset, valid_dataset, test_dataset]:
        all_texts.extend(dataset['text'])
    # トークナイザーをトレーニング (テキストのリストを使用)
    tokenizer.train_from_iterator(all_texts, trainer)
    # 特殊トークンのIDを明示的に取得
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")
    #torkenizerにpost_process属性を追加
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_token_id),
            ("[SEP]", sep_token_id),
        ],
    )
    #tokenizerにenable_padding属性を追加
    tokenizer.enable_padding(pad_id=3, pad_token="[PAD]")
    #tokenizerをセーブ
    tokenizer.save("Neural Network/LLM/Transformer/tokenizer-wiki.json")

def loadTokenizer():
    #tokenizerをリロード
    tokenizer = tokenizers.Tokenizer.from_file("Neural Network/LLM/Transformer/tokenizer-wiki.json")
    return tokenizer

def useTokenizer(tokenizer):
    #tokenizerを使う
    batch_sentences = [
    "But what about second breakfast?",
    "Don't think he knows about second breakfast, Pip.",
    "What about elevensies?",
    ]
    output1 = tokenizer.encode("Hello, y'all! How are you 😁 ?")
    output2 = tokenizer.encode("Hello, y'all!", "How are you 😁 ?")
    print(output1.tokens)
    print(output2.tokens)
    print(output2.attention_mask)
    print(output2.type_ids)
    #output2をnumpy配列に変換して結合する
    # tokensをNumPy配列に変換（文字列をそのまま使用）
    tokens_np = np.array(output2.tokens)
    # attention_maskをNumPy配列に変換
    attention_mask_np = np.array(output2.attention_mask)
    # type_idsをNumPy配列に変換
    type_ids_np = np.array(output2.type_ids)

    # 2次元配列に結合（vstackで縦方向に結合）
    combined_np_array = np.vstack([tokens_np, attention_mask_np, type_ids_np])

    # 結果を表示
    print("Combined 2D NumPy Array:")
    print(combined_np_array)

    batch_output = tokenizer.encode_batch(
        [["Hello, y'all!", "How are you 😁 ?"],
        ["Hello to you too!", "I'm fine, thank you!"]]
        )
    print("------batch_out------")
    for i in range(2):
        print(batch_output[i].tokens)
        print(batch_output[i].ids)
        print(batch_output[i].type_ids)
        print(batch_output[i].attention_mask)

def createTokenEmbeddingMatrix(tokenizer):
    #トークンIDの総数
    vocab_size = tokenizer.get_vocab_size()
    #埋め込みベクトルの次元(ハイパーパラメーター)
    embedding_dim = 768
    # 埋め込み行列をランダムに初期化 ({トークンIDの数}行{埋め込みベクトルの次元数}列の行列)
    embedding_matrix = np.random.randn(vocab_size, embedding_dim)

    return embedding_matrix

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



if __name__ == "__main__":
    # トークン数と埋め込み次元数の例
    E_len = 100  # 入力シーケンス長
    embedding_dim = 768  # 埋め込みベクトルの次元

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
    W_Q = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(2 / embedding_dim)
    W_K = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(2 / embedding_dim)
    W_V = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(2 / embedding_dim)
    # Q, K, Vの計算
    Q, K, V = calculate_QKV(embedding_matrix, W_Q, W_K, W_V)
    output, attention_weights = Multi_Head_Attention(Q, K, V, embedding_dim, 8)
    print(output)
    print(attention_weights)