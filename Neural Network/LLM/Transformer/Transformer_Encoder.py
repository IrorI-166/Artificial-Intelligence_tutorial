from datasets import load_dataset
import tokenizers
import numpy as np
#from tokenizers import Tokenizer
#from tokenizers.trainers import BpeTrainer
#from tokenizers.pre_tokenizers import Whitespace
#from tokenizers.models import BPE

#EntryToken
E = []

#データセットの用意
def createDatasets():
    # データセットをロード
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    # トレーニング/検証/テストデータセットを取得
    train_dataset = ds['train']
    valid_dataset = ds['validation']
    test_dataset = ds['test']

    return train_dataset, valid_dataset, test_dataset

#tokenizerインスタンスを作成&トレーニング
def tokenizer(train_dataset, valid_dataset, test_dataset):
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

#tokenizerをリロード
def loadTokenizer():
    tokenizer = tokenizers.Tokenizer.from_file("Neural Network/LLM/Transformer/tokenizer-wiki.json")
    return tokenizer

def createEmbeddingMatrix(tokenizer):
    #トークンIDの総数
    vocab_size = tokenizer.get_vocab_size()
    #埋め込みベクトルの次元(ハイパーパラメーター)
    embedding_dim = 768
    # 埋め込み行列をランダムに初期化 (トークンIDの数 × 埋め込みベクトルの次元数)
    embedding_matrix = np.random.randn(vocab_size, embedding_dim)

    return embedding_matrix

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