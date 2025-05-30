from flask import Flask, render_template, request, redirect, url_for, session
import json
import random
from functools import wraps
from datetime import datetime
import sys
sys.path.append('..')
from inference import Recommender

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

recommender = Recommender(
        config_path = "/home/yjchen/workspace/homework/ETEGRec/config/scientific.yaml",
        model_path="/home/yjchen/workspace/homework/ETEGRec/myckpt/scientific/May-28-2025_15-05-20270c/83.pt",
        code_path="/home/yjchen/workspace/homework/ETEGRec/myckpt/scientific/May-28-2025_15-05-20270c/83.code.json",
        rqvae_path="/home/yjchen/workspace/homework/ETEGRec/myckpt/scientific/May-28-2025_15-05-20270c/83.pt.rqvae",
        device="cpu",
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

def get_recommendations(purchase_history):
    # 使用全局的 recommender 实例
    global recommender
    
    purchased_ids = [item['id'] for item in purchase_history]
    
    recommended_ids = recommender.predict_next_item(purchased_ids)
    
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
    app.run(debug=True, port=5003)