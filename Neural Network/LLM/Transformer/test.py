import tokenizers
import numpy as np

#tokenizerをリロード
tokenizer = tokenizers.Tokenizer.from_file("Neural Network/LLM/Transformer/tokenizer-wiki.json")

#tokenizerを使う
batch_sentences = [
"But what about second breakfast?",
"Don't think he knows about second breakfast, Pip.",
"What about elevensies?",
]
output1 = tokenizer.encode("こんにちは、気分はどう？")
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