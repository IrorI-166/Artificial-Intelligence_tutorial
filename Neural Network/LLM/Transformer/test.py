from datasets import load_dataset
import tokenizers
import os
#from tokenizers import Tokenizer
#from tokenizers.trainers import BpeTrainer
#from tokenizers.pre_tokenizers import Whitespace
#from tokenizers.models import BPE

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
    #tokenizerをセーブ
    tokenizer.save("./tokenizer-wiki.json")
    #tokenizerをリロード
    tokenizer = tokenizers.Tokenizer.from_file("./tokenizer-wiki.json")
    #tokenizerを使う
    output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
    print(output.tokens)

if __name__ == "__main__":
    tokenizer()
    # データの例を表示
    for i in range(10):
        print(train_dataset[i])
        print(valid_dataset[i])
        print(test_dataset[i])