import pandas as pd
from sklearn.neighbors import NearestNeighbors

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def train_model(data):
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(data[['productId', 'rating']])
    return model

def get_recommendations(model, product_id, n_recommendations=5):
    distances, indices = model.kneighbors([[product_id, 1]], n_neighbors=n_recommendations+1)
    recommendations = [index for index in indices.flatten() if index != product_id][:n_recommendations]
    return recommendations