from datasets import load_dataset
import tokenizers
#from tokenizers import Tokenizer
#from tokenizers.trainers import BpeTrainer
#from tokenizers.pre_tokenizers import Whitespace
#from tokenizers.models import BPE
#from tokenizers.processors import TemplateProcessing
import os
import random

# すべてのテキストファイルからデータを読み込む関数
def loadTexts():
    # OpenSubtitlesのテキストファイルが格納されたフォルダのパスを指定
    root_folder = r'C:/Users/IrorI/Desktop/ProgramFiles/Datasets/OpenSubtitles'
    all_texts = []

    # 再帰的にフォルダ内のすべての.txtファイルを探索
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".txt"):  # .txtファイルのみ対象
                file_path = os.path.join(root, file)
                
                # テキストファイルを開いて内容を読み込み
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    all_texts.append(text)

    # Wikitext2 データセットのロード
    wikitext_dataset = load_dataset("wikitext", "wikitext-2-v1")

    # データセットからテキストを抽出 (train, validation, test から取得)
    for dataset in [
        wikitext_dataset['train'], wikitext_dataset['validation'], wikitext_dataset['test']
        ]:
        all_texts.extend(dataset['text'])

    # データをシャッフル
    random.shuffle(all_texts)
    
    return all_texts

#tokenizerインスタンスを作成&トレーニング
def Tokenizer(all_texts):
    #tokenizerのインスタンスを作成
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="[UNK]"))
    #trainerのインスタンスを初期化
    trainer = tokenizers.trainers.BpeTrainer(vocab_size=32000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    #tokenizerにpre_trainer属性を追加
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

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

if __name__ == "__main__":
    #テキストデータの読み込み
    all_texts = loadTexts()
    #Tokenizerのトレーニング
    Tokenizer(all_texts)