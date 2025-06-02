from flask import Flask, render_template, request, redirect, url_for, session, Response, jsonify
import json
import random
from functools import wraps
from datetime import datetime
import sys
from openai import OpenAI
import re
import os
import threading
from queue import Queue
import time

sys.path.append('..')
from inference import Recommender

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Global variables for recommendation system
recommender = Recommender(
    config_path="./config/game_llm.yaml",
    model_path="../myckpt/game/45.pt",
    code_path="../myckpt/game/45.code.json",
    rqvae_path="../myckpt/game/45.pt.rqvae",
    device="cpu",
)

# Global variables for interests
interests_queue = Queue()
interests_in_progress = False
current_interests = None
interests_lock = threading.Lock()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def load_products():
    products = []
    with open('data/game.jsonl', 'r') as f:
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
        title = product["name"] if "name" in product else None
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
        model="qwen-plus",
        messages=messages_list,
    )
    generated_text = completion.choices[0].message.content
    print("Generated interests text:", generated_text)  # Debug log
    interests_matches = re.findall(r"<interest>(.*?)</interest>", generated_text)
    print("Extracted interests:", interests_matches)  # Debug log
    return interests_matches

def interests_worker():
    global interests_in_progress, current_interests
    
    while True:
        purchase_history = interests_queue.get()
        
        with interests_lock:
            interests_in_progress = True
        
        try:
            # Extract interests
            interests = extract_interests(purchase_history)
            print("Worker extracted interests:", interests)  # Debug log
            
            with interests_lock:
                current_interests = interests
                interests_in_progress = False
                    
        except Exception as e:
            print(f"Error in interests worker: {e}")
            with interests_lock:
                interests_in_progress = False
        
        interests_queue.task_done()
        time.sleep(0.1)

# Start the interests worker thread
interests_thread = threading.Thread(target=interests_worker, daemon=True)
interests_thread.start()

def get_recommendations(purchase_history):
    global recommender, current_interests
    
    purchased_ids = [item['id'] for item in purchase_history]
    
    # Get current interests (if any)
    with interests_lock:
        interests = [current_interests] if current_interests else None
    
    # Get recommendations using current interests (might be None)
    recommended_ids = recommender.predict_next_item(purchased_ids, interests=interests)
    
    all_products = load_products()
    product_dict = {p['id']: p for p in all_products}
    
    # Filter out already purchased items from recommendations
    recommended_products = []
    for prod_id in recommended_ids:
        if prod_id in product_dict and prod_id not in purchased_ids:
            recommended_products.append(product_dict[prod_id])
    
    # If we have less than 8 recommendations, add random products
    if len(recommended_products) < 8:
        existing_ids = set(p['id'] for p in recommended_products) | set(purchased_ids)
        available_products = [p for p in all_products if p['id'] not in existing_ids]
        
        if available_products:
            num_needed = 8 - len(recommended_products)
            random_products = random.sample(available_products, min(num_needed, len(available_products)))
            recommended_products.extend(random_products)
    
    # Shuffle the final list of recommendations
    random.shuffle(recommended_products)
    
    return recommended_products

def get_random_recommendations(products, num_items=4, exclude_ids=None):
    """Get random recommendations excluding certain product IDs."""
    if exclude_ids is None:
        exclude_ids = set()
    else:
        exclude_ids = set(exclude_ids)
    
    available_products = [p for p in products if p['id'] not in exclude_ids]
    return random.sample(available_products, min(num_items, len(available_products)))

@app.route('/')
def index():
    global current_interests
    
    products = load_products()
    
    # Check if we have purchases and should show recommendations
    if 'purchases' in session and len(session['purchases']) > 0:
        # Trigger interests extraction if not in progress
        with interests_lock:
            if not interests_in_progress:
                interests_queue.put(session['purchases'])
        
        # Use existing interests for recommendation
        recommended_products = get_recommendations(session['purchases'])
        
        # Get random recommendations excluding recommended products
        exclude_ids = set(p['id'] for p in recommended_products)
        exclude_ids.update(item['id'] for item in session['purchases'])
        random_recommendations = get_random_recommendations(products, num_items=4, exclude_ids=exclude_ids)
    else:
        recommended_products = random.sample(products, min(8, len(products)))
        current_interests = None
        
        # Get different random recommendations for the featured section
        exclude_ids = set(p['id'] for p in recommended_products)
        random_recommendations = get_random_recommendations(products, num_items=4, exclude_ids=exclude_ids)
    
    return render_template('index.html', 
                         products=recommended_products,
                         random_recommendations=random_recommendations,
                         predicted_interests=current_interests)

@app.route('/product/<product_id>')
def product_detail(product_id):
    products = load_products()
    product = next((p for p in products if p['id'] == product_id), None)
    if product is None:
        return redirect(url_for('index'))
    
    # 将description列表转换为空格连接的字符串
    if 'description' in product and isinstance(product['description'], list):
        product['description'] = ' '.join(product['description'])
    
    return render_template('product_detail.html', product=product)

@app.route('/purchase', methods=['POST'])
@init_purchases
def purchase():
    product_id = request.form.get('product_id')
    product_name = request.form.get('product_name')
    product_image = request.form.get('product_image')
    product_description = request.form.get('product_description')
    
    # 如果description是字符串形式的列表，将其转换为空格连接的字符串
    if product_description and product_description.startswith('[') and product_description.endswith(']'):
        try:
            description_list = json.loads(product_description)
            if isinstance(description_list, list):
                product_description = ' '.join(description_list)
        except json.JSONDecodeError:
            pass  # 如果解析失败，保持原样
    
    # Add to purchase history
    session['purchases'].append({
        'id': product_id,
        'name': product_name,
        'image': product_image,
        'description': product_description,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    session.modified = True
    
    # Trigger interests extraction if not in progress
    with interests_lock:
        if not interests_in_progress:
            interests_queue.put(session['purchases'])
    
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
        
        # Trigger interests extraction if we have purchases and it's not in progress
        with interests_lock:
            if len(session['purchases']) > 0 and not interests_in_progress:
                interests_queue.put(session['purchases'])
            elif len(session['purchases']) == 0:
                global current_interests
                current_interests = None
    
    return redirect(url_for('purchase_history'))

@app.route('/clear_history', methods=['POST'])
@init_purchases
def clear_history():
    if 'purchases' in session:
        session['purchases'] = []
        session.modified = True
        
        # Clear current interests
        global current_interests
        with interests_lock:
            current_interests = None
    
    return redirect(url_for('purchase_history'))

@app.route('/stream')
def stream():
    def generate():
        last_interests = None
        while True:
            interests_to_send = None
            with interests_lock:
                interests_to_send = current_interests
            
            if interests_to_send != last_interests:
                if interests_to_send:
                    print("Sending interests:", interests_to_send)  # Debug log
                    data = json.dumps({
                        'interests': interests_to_send
                    })
                    yield f"data: {data}\n\n"
                last_interests = interests_to_send
            
            time.sleep(1)
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000)