import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors

class RestaurantRecommender:
    def __init__(self, restaurant_data, feature_matrix, similarity_metric='cosine'):
        """
        Initialize the recommender system
        
        Args:
            restaurant_data: DataFrame with restaurant information
            feature_matrix: DataFrame with features for similarity calculation
            similarity_metric: 'cosine', 'euclidean', or 'knn'
        """
        self.restaurant_data = restaurant_data.reset_index(drop=True)
        self.feature_matrix = feature_matrix.reset_index(drop=True)
        self.similarity_metric = similarity_metric
        
        # Create restaurant name to index mapping
        self.name_to_idx = {name: idx for idx, name in enumerate(self.restaurant_data['name'])}
        
        # Precompute similarity matrix
        self._compute_similarity_matrix()
        
    def _compute_similarity_matrix(self):
        """Compute similarity matrix based on chosen metric"""
        print(f"Computing {self.similarity_metric} similarity matrix...")
        
        if self.similarity_metric == 'cosine':
            self.similarity_matrix = cosine_similarity(self.feature_matrix)
        
        elif self.similarity_metric == 'euclidean':
            distances = euclidean_distances(self.feature_matrix)
            # Convert distances to similarities (higher = more similar)
            max_distance = distances.max()
            self.similarity_matrix = 1 - (distances / max_distance)
        
        elif self.similarity_metric == 'knn':
            # Use KNN for similarity
            self.knn_model = NearestNeighbors(
                n_neighbors=50, 
                metric='cosine', 
                algorithm='brute'
            )
            self.knn_model.fit(self.feature_matrix)
            self.similarity_matrix = None  # Will compute on-demand
        
        print(f"Similarity computation complete!")
    
    def get_restaurant_index(self, restaurant_name):
        """Get restaurant index by name"""
        if restaurant_name not in self.name_to_idx:
            # Fuzzy matching
            possible_matches = [name for name in self.name_to_idx.keys() 
                              if restaurant_name.lower() in name.lower()]
            if possible_matches:
                return self.name_to_idx[possible_matches[0]]
            else:
                raise ValueError(f"Restaurant '{restaurant_name}' not found!")
        return self.name_to_idx[restaurant_name]
    
    def get_recommendations(self, restaurant_name, n_recommendations=10, 
                          include_similar_cuisine=True, include_similar_location=True):
        """
        Get restaurant recommendations
        
        Args:
            restaurant_name: Name of the reference restaurant
            n_recommendations: Number of recommendations to return
            include_similar_cuisine: Whether to boost similar cuisine restaurants
            include_similar_location: Whether to boost same location restaurants
        """
        try:
            restaurant_idx = self.get_restaurant_index(restaurant_name)
        except ValueError as e:
            return str(e)
        
        if self.similarity_metric == 'knn':
            # Use KNN for recommendations
            distances, indices = self.knn_model.kneighbors(
                [self.feature_matrix.iloc[restaurant_idx]], 
                n_neighbors=n_recommendations+1
            )
            # Remove the restaurant itself (first result)
            recommended_indices = indices[0][1:]
            similarity_scores = 1 - distances[0][1:]  # Convert distances to similarities
        else:
            # Use precomputed similarity matrix
            similarity_scores = self.similarity_matrix[restaurant_idx]
            
            # Apply filters and boosts
            if include_similar_cuisine or include_similar_location:
                similarity_scores = self._apply_filters_and_boosts(
                    restaurant_idx, similarity_scores, 
                    include_similar_cuisine, include_similar_location
                )
            
            # Get top similar restaurants (excluding the restaurant itself)
            similar_indices = np.argsort(similarity_scores)[::-1]
            recommended_indices = [idx for idx in similar_indices if idx != restaurant_idx][:n_recommendations]
            similarity_scores = similarity_scores[recommended_indices]
        
        # Create recommendations DataFrame
        recommendations = self.restaurant_data.iloc[recommended_indices].copy()
        recommendations['similarity_score'] = similarity_scores[:len(recommended_indices)]
        
        # Select relevant columns
        columns_to_show = [
            'name', 'cuisines', 'location', 'rating', 'cost_for_two', 
            'rest_type', 'online_order', 'book_table', 'similarity_score'
        ]
        
        available_columns = [col for col in columns_to_show if col in recommendations.columns]
        recommendations = recommendations[available_columns]
        
        return recommendations
    
    def _apply_filters_and_boosts(self, restaurant_idx, similarity_scores, 
                                 include_similar_cuisine, include_similar_location):
        """Apply cuisine and location boosts to similarity scores"""
        reference_restaurant = self.restaurant_data.iloc[restaurant_idx]
        
        if include_similar_cuisine:
            # Boost restaurants with similar cuisines
            reference_cuisines = set(str(reference_restaurant['cuisines']).lower().split(', '))
            for idx in range(len(similarity_scores)):
                if idx != restaurant_idx:
                    restaurant_cuisines = set(str(self.restaurant_data.iloc[idx]['cuisines']).lower().split(', '))
                    cuisine_overlap = len(reference_cuisines.intersection(restaurant_cuisines))
                    if cuisine_overlap > 0:
                        similarity_scores[idx] *= (1 + 0.2 * cuisine_overlap)  # 20% boost per overlapping cuisine
        
        if include_similar_location:
            # Boost restaurants in the same location
            reference_location = reference_restaurant['location']
            for idx in range(len(similarity_scores)):
                if idx != restaurant_idx:
                    if self.restaurant_data.iloc[idx]['location'] == reference_location:
                        similarity_scores[idx] *= 1.3  # 30% boost for same location
        
        return similarity_scores
    
    def get_restaurant_details(self, restaurant_name):
        """Get detailed information about a restaurant"""
        try:
            restaurant_idx = self.get_restaurant_index(restaurant_name)
            return self.restaurant_data.iloc[restaurant_idx]
        except ValueError as e:
            return str(e)
    
    def find_similar_restaurants_by_criteria(self, cuisine=None, location=None, 
                                           price_range=None, rating_min=None, n_results=20):
        """Find restaurants based on specific criteria"""
        filtered_data = self.restaurant_data.copy()
        
        # Apply filters
        if cuisine:
            filtered_data = filtered_data[
                filtered_data['cuisines'].str.contains(cuisine, case=False, na=False)
            ]
        
        if location:
            filtered_data = filtered_data[
                filtered_data['location'].str.contains(location, case=False, na=False)
            ]
        
        if price_range:
            if price_range == 'budget':
                filtered_data = filtered_data[filtered_data['cost_for_two'] <= 300]
            elif price_range == 'mid':
                filtered_data = filtered_data[
                    (filtered_data['cost_for_two'] > 300) & (filtered_data['cost_for_two'] <= 600)
                ]
            elif price_range == 'expensive':
                filtered_data = filtered_data[filtered_data['cost_for_two'] > 600]
        
        if rating_min:
            filtered_data = filtered_data[filtered_data['rating'] >= rating_min]
        
        # Sort by rating and popularity
        filtered_data = filtered_data.sort_values(['rating', 'votes'], ascending=[False, False])
        
        return filtered_data.head(n_results)