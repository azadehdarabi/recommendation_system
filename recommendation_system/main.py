import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from scipy.sparse import csr_matrix

from .data_loading import extract_tags_and_categories, build_index, load_data
from .feature_engineering import compute_product_feature_vectors_parallel
from .matrix_factorization import get_svd_factors
from .recommendation_algorithms import recommend_products_device_based, \
    recommend_products_time_based, recommend_products_cbf, recommend_products_mf, recommend_popular_trending_products
from .setup import redis_client
from .user_profiles import compute_user_profiles_parallel

# Explanation templates for recommendations
EXPLANATION_TEMPLATES = {
    "mf": "Recommended because users similar to you purchased this.",
    "cbf": "Recommended because it matches your interests.",
    "popular": "Recommended because it's popular among other users.",
    "time_based": "Recommended because it's trending this season.",
    "device_based": "Recommended because it's suitable for your device.",
}


def compute_product_popularity(purchase_history, products, rating_weight=0.7, frequency_weight=0.3):
    """Precomputes product popularity based on purchase frequency and product ratings."""
    product_popularity = defaultdict(float)
    purchase_frequency = defaultdict(int)

    # Compute purchase frequency
    for purchases in purchase_history.values():
        for product_id, _, _ in purchases:
            purchase_frequency[product_id] += 1

    # Compute weighted popularity score
    for product_id in purchase_frequency:
        rating = products[product_id]["rating"]
        frequency = purchase_frequency[product_id]
        product_popularity[product_id] = (rating_weight * rating) + (frequency_weight * frequency)

    # Normalize popularity scores (optional)
    max_popularity = max(product_popularity.values(), default=1)
    for product_id in product_popularity:
        product_popularity[product_id] /= max_popularity

    return sorted(product_popularity.items(), key=lambda x: x[1], reverse=True)


def create_sparse_user_item_matrix(purchase_history, user_index, product_index):
    """Creates a sparse matrix where rows represent users and columns represent products."""
    rows, cols, data = [], [], []

    for user_id, purchases in purchase_history.items():
        for product_id, _, _ in purchases:
            rows.append(user_index[user_id])
            cols.append(product_index[product_id])
            data.append(1)

    return csr_matrix((data, (rows, cols)), shape=(len(user_index), len(product_index)))


def cache_recommendations(user_id, season_input, recommendations, ttl=3600):
    """Caches recommendations in Redis."""
    cache_key = f"recommendations:{user_id}:{','.join(season_input)}"

    try:
        redis_client.setex(cache_key, ttl, json.dumps(recommendations, ensure_ascii=False))
    except (TypeError, json.JSONDecodeError) as e:
        logging.error(f"Failed to serialize recommendations for caching: {e}")


def get_cached_recommendations(user_id, season_input):
    """Retrieves cached recommendations from Redis."""
    cache_key = f"recommendations:{user_id}:{','.join(season_input)}"
    cached_data = redis_client.get(cache_key)

    if not cached_data:
        return None
    try:
        return json.loads(cached_data)
    except json.JSONDecodeError:
        return None


def recommend_products_hybrid(user_id, season_input, users, products, contextual_signals, browsing_history,
                              purchase_history, popular_products, user_factors,
                              item_factors, user_index, product_index, user_profiles, product_feature_vectors, top_n=5):
    """Combines recommendations from multiple sources and returns the top N recommendations."""
    cached_recommendations = get_cached_recommendations(user_id, season_input)
    if cached_recommendations:
        return cached_recommendations

    is_new_user = user_id not in browsing_history and user_id not in purchase_history

    popular_recommendations = recommend_popular_trending_products(user_id, popular_products, purchase_history, top_n)

    if is_new_user:
        return [(product_id, EXPLANATION_TEMPLATES["popular"]) for product_id in popular_recommendations]

    mf_recommendations = [] if is_new_user else recommend_products_mf(user_id, user_factors, item_factors, user_index,
                                                                      product_index, purchase_history, top_n)
    cbf_recommendations = [] if is_new_user else recommend_products_cbf(user_id, user_profiles, product_feature_vectors,
                                                                        purchase_history, top_n)

    time_based_recommendations = recommend_products_time_based(season_input, products, contextual_signals)
    device_recommendations = recommend_products_device_based(user_id, users, products, purchase_history, top_n)

    if not any([mf_recommendations, cbf_recommendations, time_based_recommendations, device_recommendations]):
        return popular_recommendations[:top_n]

    mf_weight = float(os.getenv('MF_WEIGHT', 0.5))
    cbf_weight = float(os.getenv('CBF_WEIGHT', 0.4))
    popular_weight = float(os.getenv('POPULAR_WEIGHT', 0.3))
    time_base_weight = float(os.getenv('TIME_BASE_WEIGHT', 0.2))
    device_weight = float(os.getenv('DEVICE_WEIGHT', 0.2))

    if is_new_user:
        mf_weight = float(os.getenv('NEW_USER_MF_WEIGHT', 0.0))
        cbf_weight = float(os.getenv('NEW_USER_CBF_WEIGHT', 0.0))
        popular_weight = float(os.getenv('NEW_USER_POPULAR_WEIGHT', 0.5))
        time_base_weight = float(os.getenv('NEW_USER_TIME_BASE_WEIGHT', 0.3))
        device_weight = float(os.getenv('NEW_USER_DEVICE_WEIGHT', 0.2))

    recommendation_scores = defaultdict(float)
    recommendation_sources = defaultdict(list)

    for i, product_id in enumerate(mf_recommendations):
        recommendation_scores[product_id] += mf_weight * (1 - i / len(mf_recommendations))
        recommendation_sources[product_id].append("mf")

    for i, product_id in enumerate(cbf_recommendations):
        recommendation_scores[product_id] += cbf_weight * (1 - i / len(cbf_recommendations))
        recommendation_sources[product_id].append("cbf")

    for i, product_id in enumerate(popular_recommendations):
        recommendation_scores[product_id] += popular_weight * (1 - i / len(popular_recommendations))
        recommendation_sources[product_id].append("popular")

    for i, product_id in enumerate(time_based_recommendations):
        recommendation_scores[product_id] += time_base_weight * (1 - i / len(time_based_recommendations))
        recommendation_sources[product_id].append("time_based")

    for i, product_id in enumerate(device_recommendations):
        recommendation_scores[product_id] += device_weight * (1 - i / len(device_recommendations))
        recommendation_sources[product_id].append("device_based")

    sorted_recommendations = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)

    top_recommendations = []
    for product_id, _ in sorted_recommendations[:top_n]:
        sources = recommendation_sources[product_id]
        explanation = ", ".join([EXPLANATION_TEMPLATES[source] for source in sources])
        top_recommendations.append((product_id, explanation))

    cache_recommendations(user_id, season_input, top_recommendations)

    return top_recommendations


def generate_recommendations_parallel(users, season_input, products, contextual_signals, browsing_history,
                                      purchase_history, popular_products, user_factors, item_factors, user_index,
                                      product_index, user_profiles, product_feature_vectors, top_n=5):
    """Generate recommendations for all users in parallel."""
    recommendations = {}
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(recommend_products_hybrid, user_id, season_input, users, products, contextual_signals,
                            browsing_history, purchase_history, popular_products, user_factors, item_factors,
                            user_index, product_index, user_profiles, product_feature_vectors, top_n): user_id
            for user_id in users
        }
        for future in as_completed(futures):
            user_id = futures[future]
            try:
                recommendations[user_id] = future.result()
            except Exception as e:
                logging.error(f"Error generating recommendations for user {user_id}: {e}")
    return recommendations


def clear_cache_for_user(user_id):
    """Clears all cache entries for a specific user."""
    cache_keys = redis_client.keys(f"recommendations:{user_id}:*")
    for key in cache_keys:
        redis_client.delete(key)


def main():
    # Load data and precompute necessary structures
    users, products, browsing_history, purchase_history, contextual_signals = load_data()
    user_index, product_index = build_index(users, products)
    all_tags, all_categories = extract_tags_and_categories(products)
    product_feature_vectors = compute_product_feature_vectors_parallel(products, all_tags, all_categories)
    user_profiles = compute_user_profiles_parallel(users.keys(), browsing_history, purchase_history,
                                                   product_feature_vectors)
    user_item_matrix = create_sparse_user_item_matrix(purchase_history, user_index, product_index)
    user_factors, item_factors = get_svd_factors(user_item_matrix)
    popular_products = compute_product_popularity(purchase_history, products)

    season_input = ["All Year", "Summer"]  # Example season input

    if 'pytest' in sys.modules:
        for user_id in users:
            recommendations = recommend_products_hybrid(user_id, season_input, users, products, contextual_signals,
                                                        browsing_history, purchase_history,
                                                        popular_products, user_factors, item_factors, user_index,
                                                        product_index, user_profiles, product_feature_vectors)
            print(f"Recommendations for User {user_id} ({users[user_id]['name']}):")
            for product_id, explanation in recommendations:
                print(f"  - {products[product_id]['name']} (Category: {products[product_id]['category']})")
                print(f"    Explanation: {explanation}")
            print()

    else:
        recommendations = generate_recommendations_parallel(users, season_input, products, contextual_signals,
                                                            browsing_history,
                                                            purchase_history, popular_products, user_factors,
                                                            item_factors, user_index, product_index, user_profiles,
                                                            product_feature_vectors)
        for user_id, user_recommendations in recommendations.items():
            print(f"Recommendations for User {user_id} ({users[user_id]['name']}):")
            for product_id, explanation in user_recommendations:
                print(f"  - {products[product_id]['name']} (Category: {products[product_id]['category']})")
                print(f"    Explanation: {explanation}")
            print()

if __name__ == "__main__":
    main()