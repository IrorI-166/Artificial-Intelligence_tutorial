from datasets import load_dataset

# データセットをロード
ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

# トレーニング/検証/テストデータセットを取得
train_dataset = ds['train']
valid_dataset = ds['validation']
test_dataset = ds['test']

# データの例を表示
for i in range(10):
    print(train_dataset[i])
    print(valid_dataset[i])
    print(test_dataset[i])