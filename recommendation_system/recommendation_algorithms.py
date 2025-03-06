from datetime import datetime

import numpy as np
from sklearn.neighbors import NearestNeighbors


def recommend_products_mf(user_id, user_factors, item_factors, user_index, product_index, purchase_history, top_n=3):
    """Recommends products using Matrix Factorization (SVD)."""
    if user_id not in user_index:
        return []

    user_vector = user_factors[user_index[user_id]]
    scores = item_factors @ user_vector
    sorted_indices = np.argsort(scores)[::-1]

    sorted_products = [list(product_index.keys())[i] for i in sorted_indices]
    purchased_products = {p[0] for p in purchase_history.get(user_id, [])}

    return [pid for pid in sorted_products if pid not in purchased_products][:top_n]


def recommend_products_cbf(user_id, user_profiles, product_feature_vectors, purchase_history, top_n=3):
    """Recommends products using Content-Based Filtering."""
    user_profile = user_profiles[user_id]

    # Use NearestNeighbors for faster similarity search
    nn = NearestNeighbors(n_neighbors=top_n, metric="cosine")
    nn.fit(np.array(list(product_feature_vectors.values())))
    distances, indices = nn.kneighbors([user_profile])

    recommendations = []
    for idx in indices[0]:
        product_id = list(product_feature_vectors.keys())[idx]
        if product_id not in {p[0] for p in purchase_history.get(user_id, [])}:
            recommendations.append(product_id)

    return recommendations[:top_n]


def recommend_popular_trending_products(user_id, popular_products, purchase_history, top_n=3):
    """Recommends top trending products."""
    purchased_products = {p[0] for p in purchase_history.get(user_id, [])}
    return [pid for pid, _ in popular_products if pid not in purchased_products][:top_n]


def recommend_products_time_based(season_input, products, contextual_signals):
    """Recommends products based on time of day/week trends."""
    current_day = datetime.now().strftime("%A")

    trending_categories = set()
    for category, signals in contextual_signals.items():
        if current_day in signals["peak_days"] and signals["season"] in season_input:
            trending_categories.add(category)

    trending_products = []
    for product_id, product in products.items():
        if product["category"] in trending_categories:
            trending_products.append(product_id)

    return trending_products


def recommend_products_device_based(user_id, users, products, purchase_history, top_n=3):
    """Recommends products based on device type."""
    if user_id not in users:
        return []

    device_type = users[user_id]["device"]

    device_recommendations = []
    for product_id, product in products.items():
        if device_type in product["device_suitability"]:
            device_recommendations.append(product_id)

    if user_id in purchase_history:
        purchased_products = set([product_id for product_id, _, _ in purchase_history[user_id]])
        device_recommendations = [product_id for product_id in device_recommendations if
                                  product_id not in purchased_products]

    return device_recommendations[:top_n]
