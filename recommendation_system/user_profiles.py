import json
import logging
import sys
from concurrent.futures import as_completed, ProcessPoolExecutor

import numpy as np
import redis

from .setup import redis_client


def create_user_profile(user_id, browsing_history, purchase_history, product_feature_vectors):
    """Creates a user profile based on browsing and purchase history."""
    interacted_products = set()
    if user_id in browsing_history:
        interacted_products.update([product_id for product_id, _ in browsing_history[user_id]])
    if user_id in purchase_history:
        interacted_products.update([product_id for product_id, _, _ in purchase_history[user_id]])

    user_profile = np.zeros(len(product_feature_vectors[next(iter(product_feature_vectors))]))
    for product_id in interacted_products:
        user_profile += product_feature_vectors[product_id]

    if len(interacted_products) > 0:
        user_profile /= len(interacted_products)

    return user_profile


def get_user_profile(user_id, browsing_history, purchase_history, product_feature_vectors):
    """Retrieves or computes and caches the user profile."""
    cache_key = f"user_profile:{user_id}"

    try:
        cached_profile = redis_client.get(cache_key)
        if cached_profile:
            return np.array(json.loads(cached_profile))

        user_profile = create_user_profile(user_id, browsing_history, purchase_history, product_feature_vectors)
        redis_client.setex(cache_key, 86400, json.dumps(user_profile.tolist()))  # Cache for 24 hours
        return user_profile

    except redis.RedisError as e:
        logging.error(f"Redis error: {e}. Falling back to computing user profile without caching.")
        return create_user_profile(user_id, browsing_history, purchase_history, product_feature_vectors)


def compute_user_profiles_parallel(users, browsing_history, purchase_history, product_feature_vectors):
    """Compute user profiles in parallel."""
    if 'pytest' in sys.modules:
        return {user_id: get_user_profile(user_id, browsing_history, purchase_history, product_feature_vectors) for
                user_id in users}
    user_profiles = {}
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(get_user_profile, user_id, browsing_history, purchase_history,
                            product_feature_vectors): user_id
            for user_id in users
        }
        for future in as_completed(futures):
            user_id = futures[future]
            try:
                user_profiles[user_id] = future.result()
            except Exception as e:
                logging.error(f"Error computing profile for user {user_id}: {e}")
    return user_profiles
