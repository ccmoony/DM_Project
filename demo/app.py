from flask import Flask, render_template, request, redirect, url_for, session, Response
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
    config_path="./config/scientific_llm.yaml",
    model_path="../myckpt/scientific/scientific_llm/93.pt",
    code_path="../myckpt/scientific/scientific_llm/93.code.json",
    rqvae_path="../myckpt/scientific/scientific_llm/93.pt.rqvae",
    device="cpu",
)

# Add predicted interests storage
predicted_interests = None

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# Recommendation task queue and status
recommendation_queue = Queue()
current_recommendation_task = None
recommendation_in_progress = False
recommendation_result = None
recommendation_lock = threading.Lock()

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
        model="qwen-plus",
        messages=messages_list,
    )
    generated_text = completion.choices[0].message.content
    print(generated_text)
    interests_matches = re.findall(r"<interest>(.*?)</interest>", generated_text)
    interests = "\\n".join(interests_matches)

    return [interests]

def get_recommendations(purchase_history):
    global recommender, predicted_interests
    
    interests = extract_interests(purchase_history)
    predicted_interests = interests[0].split('\\n')  # Store the predicted interests
    purchased_ids = [item['id'] for item in purchase_history]
    
    recommended_ids = recommender.predict_next_item(purchased_ids, interests=interests)
    
    all_products = load_products()
    
    product_dict = {p['id']: p for p in all_products}
    
    recommended_products = []
    for prod_id in recommended_ids:
        if prod_id in product_dict and product_dict[prod_id] not in recommended_products:
            recommended_products.append(product_dict[prod_id])
    
    return recommended_products

def recommendation_worker():
    global recommendation_in_progress, recommendation_result, current_recommendation_task, predicted_interests
    
    while True:
        purchase_history = recommendation_queue.get()
        
        with recommendation_lock:
            current_recommendation_task = purchase_history
            recommendation_in_progress = True
        
        try:
            # Get recommendations
            recommendations = get_recommendations(purchase_history)
            
            with recommendation_lock:
                # Only update if this is still the current task (no newer updates)
                if current_recommendation_task == purchase_history:
                    recommendation_result = recommendations
                    recommendation_in_progress = False
                    
        except Exception as e:
            print(f"Error in recommendation worker: {e}")
            with recommendation_lock:
                recommendation_in_progress = False
        
        recommendation_queue.task_done()
        time.sleep(0.1)

# Start the recommendation worker thread
recommendation_thread = threading.Thread(target=recommendation_worker, daemon=True)
recommendation_thread.start()

@app.route('/')
def index():
    global recommendation_result, recommendation_in_progress, predicted_interests
    
    products = load_products()
    
    # Check if we have purchases and should show recommendations
    if 'purchases' in session and len(session['purchases']) > 0:
        # Check if we have a recommendation result
        with recommendation_lock:
            if recommendation_result is not None:
                displayed_products = recommendation_result
            else:
                displayed_products = random.sample(products, min(8, len(products)))
                
                # If no recommendation is in progress, start one
                if not recommendation_in_progress:
                    recommendation_queue.put(session['purchases'])
    else:
        displayed_products = random.sample(products, min(8, len(products)))
        predicted_interests = None
    
    return render_template('index.html', products=displayed_products, predicted_interests=predicted_interests)

@app.route('/purchase', methods=['POST'])
@init_purchases
def purchase():
    product_id = request.form.get('product_id')
    product_name = request.form.get('product_name')
    product_image = request.form.get('product_image')
    
    # Add to purchase history
    session['purchases'].append({
        'id': product_id,
        'name': product_name,
        'image': product_image,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    
    session.modified = True
    
    # Trigger new recommendation
    with recommendation_lock:
        global current_recommendation_task
        current_recommendation_task = session['purchases']
        recommendation_queue.put(session['purchases'])
    
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
        
        # Trigger new recommendation if we have purchases left
        if len(session['purchases']) > 0:
            with recommendation_lock:
                global current_recommendation_task
                current_recommendation_task = session['purchases']
                recommendation_queue.put(session['purchases'])
    
    return redirect(url_for('purchase_history'))

@app.route('/clear_history', methods=['POST'])
@init_purchases
def clear_history():
    if 'purchases' in session:
        session['purchases'] = []
        session.modified = True
        
        # Clear any pending recommendations
        with recommendation_lock:
            global current_recommendation_task, recommendation_result
            current_recommendation_task = None
            recommendation_result = None
    
    return redirect(url_for('purchase_history'))

@app.route('/stream')
def stream():
    def generate():
        last_interests = None
        while True:
            with recommendation_lock:
                current_interests = predicted_interests
            
            if current_interests != last_interests:
                if current_interests:
                    data = json.dumps({
                        'interests': current_interests
                    })
                    yield f"data: {data}\n\n"
                last_interests = current_interests
            
            time.sleep(1)
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, port=5000)