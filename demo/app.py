from flask import Flask, render_template, request, redirect, url_for, session
import json
import random
from functools import wraps
from datetime import datetime
import sys
from openai import OpenAI
import re
import os
sys.path.append('..')
from inference import Recommender

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

recommender = Recommender(
        config_path = "./config/scientific_llm.yaml",
        model_path="../myckpt/scientific/scientific_llm/93.pt",
        code_path="../myckpt/scientific/scientific_llm/93.code.json",
        rqvae_path="../myckpt/scientific/scientific_llm/93.pt.rqvae",
        device="cpu",
    )

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"), # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def load_products():
    products = []
    with open('data/data.jsonl', 'r') as f:
        for line in f:
            products.append(json.loads(line))
    return products
        
def init_purchases(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'purchases' not in session:
            session['purchases'] = []
        return f(*args, **kwargs)
    return decorated_function

def get_prompt(purchase_history):
    prompt = "The user's browsing history is: \n"

    for i, product in enumerate(purchase_history):

        title = product["title"] if "title" in product else None
        main_category = product["main_category"] if "main_category" in product else None
        categories = product["categories"] if "categories" in product else None
        features = product["features"] if "features" in product else None
        description = product["description"] if "description" in product else None

        prompt += f"{i + 1}. {{\n"
        prompt += f"    \"title\": \"{title}\",\n"
        prompt += f"    \"main_category\": \"{main_category}\",\n"
        prompt += f"    \"categories\": {categories},\n"
        prompt += f"    \"features\": {features},\n"
        prompt += f"    \"description\": \"{description}\"\n"
        prompt += "}\n"
        prompt += "Remember to use the <interest></interest> tags for your output.\n"

    messages = [
        {
            "role": "system", "content": "You are an expert in recommending goods. You need to predict the user's interest based on the user's browse history. Please provide exactly 5 most likely interest categories. Each line should be wrapped in <interest></interest> tags, with one interest per line inside the tags. The interests should be relevant to the user's history and not too generic. The interests should be in English."
        },
        {
            "role": "user", "content": prompt
        }
    ]

    return messages

def extract_interests(purchase_history):
    messages_list = get_prompt(purchase_history)
    completion = client.chat.completions.create(
    model="qwen-plus", # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    messages=messages_list,
    )
    generated_text = completion.choices[0].message.content
    print(generated_text)
    interests_matches = re.findall(r"<interest>(.*?)</interest>", generated_text)
    interests = "\\n".join(interests_matches)

    return [interests]

def get_recommendations(purchase_history):
    global recommender
    
    interests = extract_interests(purchase_history)
    purchased_ids = [item['id'] for item in purchase_history]
    
    recommended_ids = recommender.predict_next_item(purchased_ids, interests=interests)
    
    all_products = load_products()
    
    product_dict = {p['id']: p for p in all_products}
    
    recommended_products = []
    for prod_id in recommended_ids:
        if prod_id in product_dict and product_dict[prod_id] not in recommended_products:
            recommended_products.append(product_dict[prod_id])
    
    return recommended_products  

@app.route('/')
def index():
    products = load_products()
    if 'purchases' in session and len(session['purchases'])>0:
        displayed_products = get_recommendations(session['purchases'])
    else:
        displayed_products = random.sample(products, min(8, len(products)))
    return render_template('index.html', products=displayed_products)

@app.route('/purchase', methods=['POST'])
@init_purchases
def purchase():
    product_id = request.form.get('product_id')
    product_name = request.form.get('product_name')
    product_image = request.form.get('product_image')
    
    session['purchases'].append({
        'id': product_id,
        'name': product_name,
        'image': product_image,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    session.modified = True
    return redirect(url_for('index'))

@app.route('/purchase_history')
@init_purchases
def purchase_history():
    purchases = session.get('purchases', [])
    return render_template('purchase_history.html', purchases=purchases)

@app.route('/delete_purchase/<purchase_id>', methods=['POST'])
@init_purchases
def delete_purchase(purchase_id):
    if 'purchases' in session:
        session['purchases'] = [p for p in session['purchases'] if p['id'] != purchase_id]
        session.modified = True
    return redirect(url_for('purchase_history'))

@app.route('/clear_history', methods=['POST'])
@init_purchases
def clear_history():
    if 'purchases' in session:
        session['purchases'] = []
        session.modified = True
    return redirect(url_for('purchase_history'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)