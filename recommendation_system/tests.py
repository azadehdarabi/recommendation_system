import json

import numpy as np
import pytest
from unittest.mock import patch

from .data_loading import load_data, build_index, extract_tags_and_categories
from .feature_engineering import create_product_feature_vector
from .main import recommend_products_hybrid, create_sparse_user_item_matrix, compute_product_popularity
from .matrix_factorization import get_svd_factors
from .user_profiles import compute_user_profiles_parallel


@pytest.fixture
def data():
    """Fixture to load data."""
    return load_data()


@pytest.fixture
def indices(data):
    """Fixture to build user and product indices."""
    users, products, _, _, _ = data
    return build_index(users, products)


@pytest.fixture
def tags_and_categories(data):
    """Fixture to extract tags and categories."""
    _, products, _, _, _ = data
    return extract_tags_and_categories(products)


@pytest.fixture
def example_user_id():
    """Fixture for an example user ID."""
    return 1


@pytest.fixture
def example_season_input():
    """Fixture for an example season input."""
    return ["All Year", "Summer"]


@pytest.fixture
def new_user_id():
    """Fixture for a non-existent user ID."""
    return 999


@pytest.fixture
def empty_season_input():
    """Fixture for an empty season input."""
    return []


def test_load_data(data):
    """Test if data is loaded correctly."""
    users, products, browsing_history, purchase_history, contextual_signals = data
    assert isinstance(users, dict)
    assert isinstance(products, dict)
    assert isinstance(browsing_history, dict)
    assert isinstance(purchase_history, dict)
    assert isinstance(contextual_signals, dict)


def test_build_index(indices):
    """Test if user and product indices are built correctly."""
    user_index, product_index = indices
    assert len(user_index) == 5  # 5 users in the sample data
    assert len(product_index) == 5  # 5 products in the sample data


def test_extract_tags_and_categories(tags_and_categories):
    """Test if tags and categories are extracted correctly."""
    all_tags, all_categories = tags_and_categories
    assert isinstance(all_tags, list)
    assert isinstance(all_categories, list)
    assert len(all_tags) > 0
    assert len(all_categories) > 0


def test_create_product_feature_vector(data, tags_and_categories):
    """Test if product feature vectors are created correctly."""
    _, products, _, _, _ = data
    all_tags, all_categories = tags_and_categories
    product = products[101]  # Example product
    feature_vector = create_product_feature_vector(product, all_tags, all_categories)
    assert isinstance(feature_vector, np.ndarray)
    assert len(feature_vector) == len(all_tags) + len(all_categories)


@pytest.mark.parametrize("user_id, season_input", [
    (1, ["All Year", "Summer"]),  # Existing user with valid season input
    (999, ["All Year", "Summer"]),  # New user with valid season input
    (1, []),  # Existing user with empty season input
])
def test_recommend_products_hybrid(data, user_id, season_input):
    """Test if hybrid recommendations are generated correctly for different scenarios."""
    users, products, browsing_history, purchase_history, contextual_signals = data
    user_index, product_index = build_index(users, products)
    all_tags, all_categories = extract_tags_and_categories(products)
    product_feature_vectors = {product_id: create_product_feature_vector(product, all_tags, all_categories)
                               for product_id, product in products.items()}
    user_profiles = compute_user_profiles_parallel(users.keys(), browsing_history, purchase_history,
                                                   product_feature_vectors)
    user_item_matrix = create_sparse_user_item_matrix(purchase_history, user_index, product_index)
    user_factors, item_factors = get_svd_factors(user_item_matrix)
    popular_products = compute_product_popularity(purchase_history, products)

    recommendations = recommend_products_hybrid(
        user_id, season_input, users, products, contextual_signals, browsing_history,
        purchase_history, popular_products, user_factors, item_factors, user_index,
        product_index, user_profiles, product_feature_vectors
    )
    assert isinstance(recommendations, list)
    assert len(recommendations) <= 5  # Check if top_n=5 is respected
    for product_id, explanation in recommendations:
        assert product_id in products
        assert isinstance(explanation, str)


def test_recommend_products_hybrid_new_user(data, new_user_id, example_season_input):
    """Test hybrid recommendations for a new user."""
    users, products, browsing_history, purchase_history, contextual_signals = data
    user_index, product_index = build_index(users, products)
    all_tags, all_categories = extract_tags_and_categories(products)
    product_feature_vectors = {product_id: create_product_feature_vector(product, all_tags, all_categories)
                               for product_id, product in products.items()}
    user_profiles = compute_user_profiles_parallel(users.keys(), browsing_history, purchase_history,
                                                   product_feature_vectors)
    user_item_matrix = create_sparse_user_item_matrix(purchase_history, user_index, product_index)
    user_factors, item_factors = get_svd_factors(user_item_matrix)
    popular_products = compute_product_popularity(purchase_history, products)

    recommendations = recommend_products_hybrid(
        new_user_id, example_season_input, users, products, contextual_signals, browsing_history,
        purchase_history, popular_products, user_factors, item_factors, user_index,
        product_index, user_profiles, product_feature_vectors
    )
    assert isinstance(recommendations, list)
    assert len(recommendations) <= 5


def test_create_product_feature_vector_invalid_product(tags_and_categories):
    """Test handling of invalid product data."""
    all_tags, all_categories = tags_and_categories
    invalid_product = {"name": "Invalid Product", "tags": [], "category": "Invalid"}
    with pytest.raises(ValueError):
        create_product_feature_vector(invalid_product, all_tags, all_categories)


def test_create_product_feature_vector_missing_category(tags_and_categories):
    """Test handling of missing category in product data."""
    all_tags, all_categories = tags_and_categories
    invalid_product = {"name": "Invalid Product", "tags": ["audio"], "category": "NonExistentCategory"}
    with pytest.raises(ValueError):
        create_product_feature_vector(invalid_product, all_tags, all_categories)


def test_recommend_products_hybrid_no_data(data, example_user_id, empty_season_input):
    """Test hybrid recommendations with no data."""
    users, products, browsing_history, purchase_history, contextual_signals = data
    user_index, product_index = build_index(users, products)
    all_tags, all_categories = extract_tags_and_categories(products)
    product_feature_vectors = {product_id: create_product_feature_vector(product, all_tags, all_categories)
                               for product_id, product in products.items()}
    user_profiles = compute_user_profiles_parallel(users.keys(), browsing_history, purchase_history,
                                                   product_feature_vectors)
    user_item_matrix = create_sparse_user_item_matrix(purchase_history, user_index, product_index)
    user_factors, item_factors = get_svd_factors(user_item_matrix)
    popular_products = compute_product_popularity(purchase_history, products)

    recommendations = recommend_products_hybrid(
        example_user_id, empty_season_input, users, products, contextual_signals, browsing_history,
        purchase_history, popular_products, user_factors, item_factors, user_index,
        product_index, user_profiles, product_feature_vectors
    )
    assert isinstance(recommendations, list)
    assert len(recommendations) <= 5


def test_recommend_products_hybrid_specific_recommendations(data, example_user_id, example_season_input):
    """Test if specific recommendations are generated for a user."""
    users, products, browsing_history, purchase_history, contextual_signals = data
    user_index, product_index = build_index(users, products)
    all_tags, all_categories = extract_tags_and_categories(products)
    product_feature_vectors = {product_id: create_product_feature_vector(product, all_tags, all_categories)
                               for product_id, product in products.items()}
    user_profiles = compute_user_profiles_parallel(users.keys(), browsing_history, purchase_history,
                                                   product_feature_vectors)
    user_item_matrix = create_sparse_user_item_matrix(purchase_history, user_index, product_index)
    user_factors, item_factors = get_svd_factors(user_item_matrix)
    popular_products = compute_product_popularity(purchase_history, products)

    recommendations = recommend_products_hybrid(
        example_user_id, example_season_input, users, products, contextual_signals, browsing_history,
        purchase_history, popular_products, user_factors, item_factors, user_index,
        product_index, user_profiles, product_feature_vectors
    )
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0  # Ensure at least one recommendation is generated

    # Check if specific products are recommended
    recommended_product_ids = [product_id for product_id, _ in recommendations]
    assert 102 in recommended_product_ids


@patch("recommendation_system.main.redis_client.get")
def test_recommend_products_hybrid_cached(mock_redis_get, data, example_user_id, example_season_input):
    """Test if cached recommendations are retrieved correctly."""
    mock_redis_get.return_value = json.dumps(([1.0, 2.0], [3.0, 4.0]))

    users, products, browsing_history, purchase_history, contextual_signals = data
    user_index, product_index = build_index(users, products)
    all_tags, all_categories = extract_tags_and_categories(products)
    product_feature_vectors = {product_id: create_product_feature_vector(product, all_tags, all_categories)
                               for product_id, product in products.items()}
    user_profiles = compute_user_profiles_parallel(users.keys(), browsing_history, purchase_history,
                                                   product_feature_vectors)
    user_item_matrix = create_sparse_user_item_matrix(purchase_history, user_index, product_index)
    user_factors, item_factors = get_svd_factors(user_item_matrix)
    popular_products = compute_product_popularity(purchase_history, products)

    recommendations = recommend_products_hybrid(
        example_user_id, example_season_input, users, products, contextual_signals, browsing_history,
        purchase_history, popular_products, user_factors, item_factors, user_index,
        product_index, user_profiles, product_feature_vectors
    )
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0  # Ensure recommendations are generated
