import os
import requests
from dotenv import load_dotenv
import json

# 環境変数を読み込む
load_dotenv()

# APIキーを環境変数から取得
API_KEY = os.getenv("API_KEY")
print(f"API_KEY: {API_KEY}")

# プロンプトからレシピを提案する関数
def chatgpt(prompt):
    # エンドポイントへのPOSTリクエストのヘッダ
    header = {
        "Content-Type" : "application/json", # ボディのコンテンツタイプを指定
        "Authorization" : f"Bearer {API_KEY}",
    }

    # JSONデータを正しくエンコードするためにjson.dumpsを使用
    body = json.dumps({
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "top_p": 0,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "seed": 42,
    })

    # エンドポイントからのPOSTレスポンス
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=header, data=body.encode('utf_8'))
    rj = response.json() # デシリアライズ
    answer = rj["choices"][0]["message"]["content"]

    # 各レシピを分割してリストに格納
    recipes = answer.split('\n\n')

    # 各レシピの名前、材料、手順を辞書に格納
    recipe_details = []
    for recipe in recipes:
        lines = recipe.split('\n')
        recipe_name = lines[0].strip('・')

        ingredients_start_index = None
        for index, item in enumerate(lines):
            if '材料' in item:
                ingredients_start_index = index
                break
        instructions_start_index = None
        for index, item in enumerate(lines):
            if '手順' in item:
                instructions_start_index = index
                break

        ingredients = lines[ingredients_start_index : instructions_start_index] 
        procedure = lines[instructions_start_index:]

        ingredients = '\n'.join(line.strip() for line in ingredients)
        procedure = '\n'.join(line.strip() for line in procedure)

        recipe_details.append({
            'name': recipe_name,
            'ingredients': ingredients,
            'procedure': procedure
        })

    return recipe_details

# デバッグ用
if __name__ == "__main__":
    prompt = f"りんご、はちみつ、バナナを使った3つのレシピの料理名と材料と手順を挙げてください。材料の分量を必ず教えてください。\
        出力のフォーマット：\
            ・料理名1\
                材料\
                手順\
            ・料理名2\
                材料\
                手順\
            ・料理名3\
                材料\
                手順\
            "

    recipes = chatgpt(prompt)
    # print(recipes)
