# Recommendation System

This project implements a **hybrid recommendation system** that combines multiple recommendation algorithms to provide personalized product recommendations to users. It uses collaborative filtering, content-based filtering, matrix factorization, and contextual signals to generate recommendations.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Redis server (for caching recommendations)

### Steps

1. **Clone the Repository**:
    ```bash
     git clone https://github.com/your-username/recommendation-system.git
     cd recommendation-system
   ```
   
2. **Set Up a Virtual Environment**:
    ```bash
     python -m venv venv
     source venv/bin/activate 
    ```
   
3. **Install Dependencies**:
    ```bash
     pip install -r requirements.txt
    ```
   
4. **Set Up Environment Variables**:
    - Copy the .env.sample file to .env:
    ```bash
     cp .env.sample .env
    ```
    - Update the .env file with your Redis configuration and recommendation weights.

5. **Run Redis**:
    - Ensure Redis is running on your machine. You can start it using:
    ```bash
     redis-server
    ```

6. **Run the Recommendation System**:
    ```bash
     python main.py
    ```
   

# Overview of the Recommendation Algorithm

The recommendation system combines the following components to generate personalized recommendations:

1. **Matrix Factorization (Collaborative Filtering)**
   - Uses user-item interaction data to identify latent factors and predict user preferences.

   - Suitable for users with a history of interactions.

2. **Content-Based Filtering**

   - Recommends products based on user preferences and product features (e.g., tags, categories).

   - Ideal for new users or when interaction data is sparse.

3. **Popular and Trending Products**

   - Recommends products that are popular or trending among all users.

   - Useful for new users or as a fallback when other methods don't provide enough recommendations.

4. **Contextual Signals**

   - Considers seasonal and device-based signals to tailor recommendations.

   - For example, recommends winter clothing during winter or mobile-friendly products for mobile users.

5. **Hybrid Approach**

   - Combines the above methods using weighted scores to generate final recommendations.

   - Weights for each method can be configured via environment variables.

For detailed documentation, refer to the docs/ directory.