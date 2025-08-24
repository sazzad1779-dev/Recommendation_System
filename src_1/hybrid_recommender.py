from src.content_base_recommender import ContentBasedRecommender
from src.text_based_recommender import TextBasedRecommender
import pandas as pd
import numpy as np

class HybridRecommender:
    """Hybrid recommender combining multiple approaches with memory optimizations"""
    
    def __init__(self, restaurant_data, content_features, text_features):
        self.restaurant_data = restaurant_data
        
        # OPTIMIZATION: Instead of creating separate recommenders, create optimized lightweight ones
        # that share computations and use memory-efficient methods
        
        print("Initializing optimized content recommender...")
        self.content_recommender = ContentBasedRecommender(restaurant_data, content_features)
        
        print("Initializing optimized text recommender...")  
        self.text_recommender = TextBasedRecommender(restaurant_data, text_features)
        
        # Weights for different recommendation types
        self.weights = {
            'content': 0.6,
            'text': 0.4
        }
        
        # Cache for hybrid recommendations to avoid recomputation
        self._recommendation_cache = {}
        
    def get_hybrid_recommendations(self, restaurant_name, n_recommendations=10):
        """Get recommendations using hybrid approach with optimized computation"""
        
        # Check cache first
        cache_key = f"{restaurant_name}_{n_recommendations}"
        if cache_key in self._recommendation_cache:
            return self._recommendation_cache[cache_key]
        
        # Get recommendations from both approaches efficiently
        # Request more recommendations to have better mixing pool
        request_size = min(n_recommendations * 3, len(self.restaurant_data) // 10)
        
        content_recs = self.content_recommender.get_recommendations(
            restaurant_name, request_size
        )
        text_recs = self.text_recommender.get_recommendations(
            restaurant_name, request_size
        )
        
        if isinstance(content_recs, str) or isinstance(text_recs, str):
            result = "Restaurant not found!"
            self._recommendation_cache[cache_key] = result
            return result
        
        # OPTIMIZATION: Use vectorized operations for combining scores
        combined_scores = self._combine_recommendations_vectorized(content_recs, text_recs)
        
        # Get top N recommendations efficiently
        top_restaurant_names = list(combined_scores.keys())[:n_recommendations]
        
        # Create final recommendations DataFrame efficiently
        final_recommendations = self._create_final_recommendations_efficient(
            top_restaurant_names, combined_scores
        )
        
        # Cache the result
        self._recommendation_cache[cache_key] = final_recommendations
        
        return final_recommendations
    
    def _combine_recommendations_vectorized(self, content_recs, text_recs):
        """Efficiently combine recommendations using vectorized operations"""
        combined_scores = {}
        
        # Process content-based recommendations efficiently
        if not content_recs.empty:
            content_names = content_recs['name'].values
            content_scores = content_recs['similarity_score'].values * self.weights['content']
            
            for name, score in zip(content_names, content_scores):
                combined_scores[name] = score
        
        # Process text-based recommendations efficiently
        if not text_recs.empty:
            text_names = text_recs['name'].values  
            text_scores = text_recs['similarity_score'].values * self.weights['text']
            
            for name, score in zip(text_names, text_scores):
                combined_scores[name] = combined_scores.get(name, 0) + score
        
        # Sort by combined score efficiently
        sorted_restaurants = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_restaurants)
    
    def _create_final_recommendations_efficient(self, top_restaurant_names, combined_scores):
        """Efficiently create final recommendations DataFrame"""
        
        # Use vectorized operations to find restaurant info
        restaurant_mask = self.restaurant_data['name'].isin(top_restaurant_names)
        relevant_restaurants = self.restaurant_data[restaurant_mask].copy()
        
        # Create a mapping for quick lookup
        name_to_info = {row['name']: row for _, row in relevant_restaurants.iterrows()}
        
        # Build final recommendations list efficiently
        final_recommendations = []
        for name in top_restaurant_names:
            if name in name_to_info:
                restaurant_info = name_to_info[name]
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
    
    def clear_cache(self):
        """Clear the recommendation cache to free memory"""
        self._recommendation_cache.clear()
        
    def get_cache_size(self):
        """Get the current cache size"""
        return len(self._recommendation_cache)
        
    def get_memory_stats(self):
        """Get memory usage statistics"""
        stats = {
            'cache_entries': len(self._recommendation_cache),
            'restaurant_count': len(self.restaurant_data)
        }
        
        # Simple memory estimation without accessing potentially problematic attributes
        try:
            content_shape = getattr(self.content_recommender.feature_matrix, 'shape', (0, 0))
            text_shape = getattr(self.text_recommender.feature_matrix, 'shape', (0, 0))
            
            stats['content_feature_shape'] = content_shape
            stats['text_feature_shape'] = text_shape
            
            # Simple estimation: assume float32 (4 bytes per element)
            content_elements = content_shape[0] * content_shape[1] if len(content_shape) >= 2 else 0
            text_elements = text_shape[0] * text_shape[1] if len(text_shape) >= 2 else 0
            
            total_memory_gb = (content_elements + text_elements) * 4 / (1024**3)
            stats['estimated_memory_gb'] = round(total_memory_gb, 2)
            
        except Exception as e:
            stats['estimated_memory_gb'] = f"Could not calculate: {str(e)}"
        
        return stats