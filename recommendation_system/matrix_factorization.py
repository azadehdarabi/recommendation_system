import json
import logging

import numpy as np
import redis
from scipy.sparse.linalg import svds

from .setup import redis_client


def perform_svd(user_item_matrix, k=2):
    """Performs Singular Value Decomposition (SVD) on the user-item matrix."""
    k = min(k, min(user_item_matrix.shape) - 1)
    if k <= 0:
        raise ValueError("Invalid SVD dimension.")

    try:
        U, sigma, Vt = svds(user_item_matrix, k=k)
        return U @ np.diag(sigma), Vt.T
    except Exception as e:
        logging.error(f"SVD computation failed: {e}. Returning empty factors.")
        return np.zeros((user_item_matrix.shape[0], k)), np.zeros((user_item_matrix.shape[1], k))


def get_svd_factors(user_item_matrix, k=2):
    """Retrieves or computes and caches SVD factors."""
    cache_key = f"svd_factors:{user_item_matrix.shape[0]}:{user_item_matrix.shape[1]}"

    try:
        cached_factors = redis_client.get(cache_key)

        if cached_factors:
            user_factors, item_factors = json.loads(cached_factors)
            return np.array(user_factors), np.array(item_factors)

        user_factors, item_factors = perform_svd(user_item_matrix, k)
        redis_client.setex(cache_key, 86400,
                           json.dumps((user_factors.tolist(), item_factors.tolist())))  # Cache for 24 hours
        return user_factors, item_factors
    except redis.RedisError as e:
        logging.error(f"Redis error: {e}. Falling back to computing SVD factors without caching.")
        return perform_svd(user_item_matrix, k)