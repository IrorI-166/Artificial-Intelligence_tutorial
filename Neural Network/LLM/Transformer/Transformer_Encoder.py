from datasets import load_dataset
import tokenizers
#from tokenizers import Tokenizer
#from tokenizers.trainers import BpeTrainer
#from tokenizers.pre_tokenizers import Whitespace
#from tokenizers.models import BPE

#EntryToken
E = []

#データセットの用意
def createDatasets():
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
    #データセットでtokenizerをトレーニング
    files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
    tokenizer.train(files, trainer)
    #tokenizerをセーブ
    tokenizer.save("./tokenizer-wiki.json")
    #tokenizerをリロード
    tokenizer = tokenizers.Tokenizer.from_file("./tokenizer-wiki.json")
    #tokenizerを使う
    output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
    print(output.token)