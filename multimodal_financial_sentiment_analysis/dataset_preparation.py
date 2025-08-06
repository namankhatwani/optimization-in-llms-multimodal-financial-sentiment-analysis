!pip install googletrans==4.0.0-rc1
# Loading instruction_subset.json
import json
instruction_path = "C:/Users/Naman/Downloads/instruction_subset.json"

with open(instruction_path, 'r', encoding='utf-8') as f:
    instruction_data = json.load(f)

# first item
print(json.dumps(instruction_data[0], indent=2, ensure_ascii=False))
from googletrans import Translator
from tqdm import tqdm

translator = Translator()

def translate_json(data):
    for item in tqdm(data):
        for conv in item.get("conversations", []):
            try:
                original_text = conv["value"]
                translated = translator.translate(original_text, src='zh-cn', dest='en')
                conv["value"] = translated.text
            except Exception as e:
                print(f"Error translating ID {item['id']}: {e}")
    return data
translated_instruction_data = translate_json(instruction_data)
output_path = "file_path"

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(translated_instruction_data, f, ensure_ascii=False, indent=2)


