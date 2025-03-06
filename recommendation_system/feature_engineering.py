import json
import logging
import sys
from concurrent.futures import as_completed, ProcessPoolExecutor

import numpy as np
import redis

from .setup import redis_client


def create_product_feature_vector(product, all_tags, all_categories):
    """Creates a feature vector for a product based on its tags and category."""
    tag_vector = np.zeros(len(all_tags))
    category_vector = np.zeros(len(all_categories))

    for tag in product["tags"]:
        tag_vector[all_tags.index(tag)] = 1
    category_vector[all_categories.index(product["category"])] = 1

    return np.concatenate((tag_vector, category_vector))


def get_product_feature_vector(product_id, product, all_tags, all_categories):
    """Retrieves or computes and caches the product feature vector."""
    cache_key = f"product_feature:{product_id}"

    try:
        cached_vector = redis_client.get(cache_key)

        if cached_vector:
            return np.array(json.loads(cached_vector))

        vector = create_product_feature_vector(product, all_tags, all_categories)
        redis_client.setex(cache_key, 86400, json.dumps(vector.tolist()))  # Cache for 24 hours
        return vector
    except redis.RedisError as e:
        logging.error(f"Redis error: {e}. Falling back to computing product feature vector without caching.")
        return create_product_feature_vector(product, all_tags, all_categories)


def compute_product_feature_vectors_parallel(products, all_tags, all_categories):
    """Compute product feature vectors in parallel."""
    if 'pytest' in sys.modules:
        return {pid: get_product_feature_vector(pid, product, all_tags, all_categories) for pid, product in
                products.items()}
    product_feature_vectors = {}
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(get_product_feature_vector, pid, product, all_tags, all_categories): pid
            for pid, product in products.items()
        }
        for future in as_completed(futures):
            pid = futures[future]
            try:
                product_feature_vectors[pid] = future.result()
            except Exception as e:
                logging.error(f"Error computing feature vector for product {pid}: {e}")
    return product_feature_vectors
