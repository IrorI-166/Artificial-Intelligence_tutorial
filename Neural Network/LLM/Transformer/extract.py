import os
from xml.etree import ElementTree as ET

# ルートフォルダ（年ごとのフォルダが入っているフォルダ）
root_folder = 'C:/Users/IrorI/Desktop/ProgramFiles/Datasets/ja/OpenSubtitles/xml/ja'

# 出力フォルダ
output_folder = 'C:/Users/IrorI/Desktop/ProgramFiles/Datasets/OpenSubtitles'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 再帰的にすべてのXMLファイルを処理する
for root, dirs, files in os.walk(root_folder):
    for file in files:
        if file.endswith(".xml"):  # XMLファイルのみを対象とする
            file_path = os.path.join(root, file)

            # XMLファイルをパースしてテキストを抽出
            try:
                tree = ET.parse(file_path)
                root_xml = tree.getroot()

                # テキストの抽出（<w>タグの内容を対象とする）
                text_data = []
                for s in root_xml.findall('.//s'):
                    sentence = ''.join([w.text for w in s.findall('.//w')])
                    text_data.append(sentence)

                # 抽出したテキストを改行で結合
                extracted_text = '\n'.join(text_data)

                # 抽出したテキストを新しいテキストファイルとして保存
                relative_path = os.path.relpath(file_path, root_folder)  # 相対パスを取得
                output_file_path = os.path.join(output_folder, relative_path.replace(".xml", ".txt"))

                # 出力先のディレクトリが存在しない場合は作成
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                # テキストファイルとして保存
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)

                print(f"Processed: {file_path}")

            except ET.ParseError as e:
                print(f"Error parsing {file_path}: {e}")
