from src.recommender import RestaurantRecommender

class ContentBasedRecommender(RestaurantRecommender):
    """Recommender based on restaurant content features (cuisine, location, price, etc.)"""
    
    def __init__(self, restaurant_data, feature_matrix):
        super().__init__(restaurant_data, feature_matrix, similarity_metric='cosine')
        
    def get_cuisine_based_recommendations(self, restaurant_name, n_recommendations=10):
        """Get recommendations based primarily on cuisine similarity"""
        return self.get_recommendations(
            restaurant_name, n_recommendations, 
            include_similar_cuisine=True, include_similar_location=False
        )
    
    def get_location_based_recommendations(self, restaurant_name, n_recommendations=10):
        """Get recommendations based primarily on location similarity"""
        return self.get_recommendations(
            restaurant_name, n_recommendations,
            include_similar_cuisine=False, include_similar_location=True
        )