from src.recommender import RestaurantRecommender

class TextBasedRecommender(RestaurantRecommender):
    """Recommender based on text features (reviews, descriptions, etc.)"""
    
    def __init__(self, restaurant_data, feature_matrix):
        # Use KNN for text-based recommendations as it's more memory efficient for high-dimensional text features
        super().__init__(restaurant_data, feature_matrix, similarity_metric='knn')
        
    def get_text_similarity_recommendations(self, restaurant_name, n_recommendations=10):
        """Get recommendations based on text similarity with enhanced text-specific logic"""
        return self.get_recommendations(
            restaurant_name, n_recommendations, 
            include_similar_cuisine=False, include_similar_location=False
        )
    
    def get_cuisine_aware_text_recommendations(self, restaurant_name, n_recommendations=10):
        """Get text-based recommendations with cuisine awareness"""
        return self.get_recommendations(
            restaurant_name, n_recommendations, 
            include_similar_cuisine=True, include_similar_location=False
        )