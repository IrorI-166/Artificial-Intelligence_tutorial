from tokenizers import Tokenizer
from datasets import load_dataset
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE

files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
tokenizer.train(files, trainer)

tokenizer.save("./tokenizer-wiki.json")
tokenizer = Tokenizer.from_file("./tokenizer-wiki.json")
output = tokenizer.encode("Hello, y'all! How are you 😁 ?")

#EntryToken
E = []

#データセットの用意
def createDatasets():
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    # トレーニング/検証/テストデータセットを取得
    train_dataset = ds['train']
    valid_dataset = ds['validation']
    test_dataset = ds['test']

#tokenizerのインスタンスを作成
def torkenizer():
    #tokenizerのインスタンスを作成
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    #trainerのインスタンスを初期化
    trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    #tokenizerにpre_trainer属性を追加
    tokenizer.pre_tokenizer = Whitespace()