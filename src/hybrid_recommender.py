from src.content_base_recommender import ContentBasedRecommender
from src.text_based_recommender import TextBasedRecommender
import pandas as pd
class HybridRecommender:
    """Hybrid recommender combining multiple approaches"""
    
    def __init__(self, restaurant_data, content_features, text_features):
        self.restaurant_data = restaurant_data
        
        # Initialize individual recommenders
        self.content_recommender = ContentBasedRecommender(restaurant_data, content_features)
        self.text_recommender = TextBasedRecommender(restaurant_data, text_features)
        
        # Weights for different recommendation types
        self.weights = {
            'content': 0.6,
            'text': 0.4
        }
    
    def get_hybrid_recommendations(self, restaurant_name, n_recommendations=10):
        """Get recommendations using hybrid approach"""
        
        # Get recommendations from both approaches
        content_recs = self.content_recommender.get_recommendations(
            restaurant_name, n_recommendations * 2
        )
        text_recs = self.text_recommender.get_recommendations(
            restaurant_name, n_recommendations * 2
        )
        
        if isinstance(content_recs, str) or isinstance(text_recs, str):
            return "Restaurant not found!"
        
        # Combine and weight scores
        combined_scores = {}
        
        # Process content-based recommendations
        for idx, row in content_recs.iterrows():
            restaurant_name_rec = row['name']
            score = row['similarity_score'] * self.weights['content']
            combined_scores[restaurant_name_rec] = combined_scores.get(restaurant_name_rec, 0) + score
        
        # Process text-based recommendations
        for idx, row in text_recs.iterrows():
            restaurant_name_rec = row['name']
            score = row['similarity_score'] * self.weights['text']
            combined_scores[restaurant_name_rec] = combined_scores.get(restaurant_name_rec, 0) + score
        
        # Sort by combined score
        sorted_restaurants = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        top_restaurant_names = [name for name, score in sorted_restaurants[:n_recommendations]]
        
        # Create final recommendations DataFrame
        final_recommendations = []
        for name in top_restaurant_names:
            restaurant_info = self.restaurant_data[self.restaurant_data['name'] == name].iloc[0]
            final_recommendations.append({
                'name': restaurant_info['name'],
                'cuisines': restaurant_info['cuisines'],
                'location': restaurant_info['location'],
                'rating': restaurant_info['rating'],
                'cost_for_two': restaurant_info['cost_for_two'],
                'rest_type': restaurant_info['rest_type'],
                'online_order': restaurant_info['online_order'],
                'book_table': restaurant_info['book_table'],
                'hybrid_score': combined_scores[name]
            })
        
        return pd.DataFrame(final_recommendations)