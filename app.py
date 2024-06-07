from flask import Flask, request, jsonify
from model import load_data, train_model, get_recommendations

app = Flask(__name__)

# Cargar y entrenar el modelo
data = load_data('data/fake_data.csv')
model = train_model(data)

@app.route('/recommend', methods=['GET'])
def recommend():
    product_id = int(request.args.get('productId'))
    recommendations = get_recommendations(model, product_id)
    return jsonify({
        'productId': product_id,
        'recommendations': recommendations
    })

if __name__ == '__main__':
    app.run(debug=True)