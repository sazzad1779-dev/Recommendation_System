from src.recommender import RestaurantRecommender
class TextBasedRecommender(RestaurantRecommender):
    """Recommender based on text features (reviews, dishes, descriptions)"""
    
    def __init__(self, restaurant_data, feature_matrix):
        super().__init__(restaurant_data, feature_matrix, similarity_metric='cosine')