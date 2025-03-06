def load_data():
    """Loads sample data for users, products, browsing history, purchase history, and contextual signals."""
    users = {
        1: {"name": "Alice", "location": "New York", "device": "mobile"},
        2: {"name": "Bob", "location": "Los Angeles", "device": "desktop"},
        3: {"name": "Charlie", "location": "Chicago", "device": "mobile"},
        4: {"name": "Diana", "location": "San Francisco", "device": "desktop"},
        5: {"name": "Mary", "location": "San Francisco", "device": "desktop"},
    }

    products = {
        101: {"name": "Wireless Earbuds", "category": "Electronics", "tags": ["audio", "wireless", "Bluetooth"],
              "rating": 4.5, "device_suitability": ["mobile", "tablet"]},
        102: {"name": "Smartphone Case", "category": "Accessories", "tags": ["phone", "protection", "case"],
              "rating": 4.2,
              "device_suitability": ["mobile"]},
        103: {"name": "Yoga Mat", "category": "Fitness", "tags": ["exercise", "mat", "yoga"], "rating": 4.7,
              "device_suitability": ["tablet"]},
        104: {"name": "Electric Toothbrush", "category": "Personal Care", "tags": ["hygiene", "electric", "toothbrush"],
              "rating": 4.3, "device_suitability": ["mobile"]},
        105: {"name": "Laptop Stand", "category": "Office Supplies", "tags": ["work", "laptop", "stand"], "rating": 4.6,
              "device_suitability": ["desktop"]},
    }

    browsing_history = {
        1: [(101, "2025-03-04 10:00:00"), (103, "2023-10-01 10:05:00")],
        2: [(102, "2025-03-04 11:30:00")],
        3: [(104, "2025-03-04 14:30:00")],
        4: [(105, "2025-03-04 16:30:00")],
    }

    purchase_history = {
        1: [(101, 1, "2025-03-04 10:00:00")],
        2: [(105, 2, "2025-03-04 12:00:00")],
        3: [(103, 1, "2025-03-04 12:00:00")],
        4: [(101, 1, "2025-03-04 12:00:00")],
    }

    contextual_signals = {
        "Electronics": {"peak_days": ["Friday", "Saturday"], "season": "Holiday"},
        "Fitness": {"peak_days": ["Monday", "Wednesday"], "season": "Summer"},
        "Office Supplies": {"peak_days": ["Tuesday", "Thursday"], "season": "Back-to-School"},
        "Personal Care": {"peak_days": ["Sunday"], "season": "All Year"},
    }
    return users, products, browsing_history, purchase_history, contextual_signals


def build_index(users, products):
    """Builds index mappings for users and products."""
    user_index = {uid: idx for idx, uid in enumerate(users.keys())}
    product_index = {pid: idx for idx, pid in enumerate(products.keys())}
    return user_index, product_index


def extract_tags_and_categories(products):
    """Extracts unique product tags and categories."""
    all_tags = {tag for product in products.values() for tag in product["tags"]}
    all_categories = {product["category"] for product in products.values()}
    return list(all_tags), list(all_categories)