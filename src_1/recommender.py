import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

class RestaurantRecommender:
    def __init__(self, restaurant_data, feature_matrix, similarity_metric='cosine'):
        """
        Initialize the recommender system with optimizations
        
        Args:
            restaurant_data: DataFrame with restaurant information
            feature_matrix: DataFrame with features for similarity calculation
            similarity_metric: 'cosine', 'euclidean', or 'knn'
        """
        self.restaurant_data = restaurant_data.reset_index(drop=True)
        self.similarity_metric = similarity_metric
        
        # Optimize feature matrix storage
        self.feature_matrix = self._optimize_feature_matrix(feature_matrix)
        
        # Create restaurant name to index mapping
        self.name_to_idx = {name: idx for idx, name in enumerate(self.restaurant_data['name'])}
        
        # DON'T precompute full similarity matrix - compute on demand instead
        self.similarity_matrix = None
        
        # Initialize similarity computation method
        self._initialize_similarity_method()
        
    def _optimize_feature_matrix(self, feature_matrix):
        """Optimize feature matrix storage"""
        # Convert to numpy array with float32 to save memory
        if hasattr(feature_matrix, 'values'):  # pandas DataFrame
            feature_array = feature_matrix.values.astype(np.float32)
        else:  # already numpy array
            feature_array = feature_matrix.astype(np.float32)
        
        # Check if sparse representation would be beneficial
        sparsity = (feature_array == 0).sum() / feature_array.size
        if sparsity > 0.5:  # If more than 50% zeros, use sparse
            print(f"Using sparse matrix representation (sparsity: {sparsity:.2%})")
            return csr_matrix(feature_array)
        else:
            print(f"Using dense matrix with float32 (sparsity: {sparsity:.2%})")
            return feature_array
    
    def _initialize_similarity_method(self):
        """Initialize similarity computation method"""
        print(f"Initializing {self.similarity_metric} similarity computation...")
        
        if self.similarity_metric == 'knn':
            # Use KNN for memory-efficient similarity
            self.knn_model = NearestNeighbors(
                n_neighbors=min(200, len(self.restaurant_data) // 5), 
                metric='cosine', 
                algorithm='brute'
            )
            self.knn_model.fit(self.feature_matrix)
        
        print(f"Similarity method initialized!")
    
    def _compute_similarity_for_restaurant(self, restaurant_idx):
        """Compute similarity for a single restaurant with caching"""
        # Convert restaurant_idx to int to make it hashable for caching
        cache_key = int(restaurant_idx)
        
        # Check if we have cached this computation
        if hasattr(self, '_similarity_cache') and cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        # Initialize cache if it doesn't exist
        if not hasattr(self, '_similarity_cache'):
            self._similarity_cache = {}
        
        if self.similarity_metric == 'cosine':
            if hasattr(self.feature_matrix, 'toarray'):  # Sparse matrix
                similarities = cosine_similarity(
                    self.feature_matrix[restaurant_idx:restaurant_idx+1], 
                    self.feature_matrix
                ).flatten()
            else:  # Dense matrix (numpy array)
                similarities = cosine_similarity(
                    self.feature_matrix[restaurant_idx:restaurant_idx+1], 
                    self.feature_matrix
                ).flatten()
        
        elif self.similarity_metric == 'euclidean':
            if hasattr(self.feature_matrix, 'toarray'):  # Sparse matrix
                distances = euclidean_distances(
                    self.feature_matrix[restaurant_idx:restaurant_idx+1], 
                    self.feature_matrix
                ).flatten()
            else:  # Dense matrix
                distances = euclidean_distances(
                    self.feature_matrix[restaurant_idx:restaurant_idx+1], 
                    self.feature_matrix
                ).flatten()
            # Convert distances to similarities
            max_distance = distances.max()
            similarities = 1 - (distances / max_distance) if max_distance > 0 else np.ones_like(distances)
        
        elif self.similarity_metric == 'knn':
            # Use KNN for efficient computation
            feature_row = self.feature_matrix[restaurant_idx]
            if hasattr(self.feature_matrix, 'toarray'):  # Sparse matrix
                feature_row = feature_row.toarray()
            
            distances, indices = self.knn_model.kneighbors(
                [feature_row], 
                n_neighbors=min(200, len(self.restaurant_data))
            )
            # Create full similarity array
            similarities = np.zeros(len(self.restaurant_data), dtype=np.float32)
            similarities[indices[0]] = 1 - distances[0]  # Convert distances to similarities
        
        # Cache the result (limit cache size to avoid memory issues)
        if len(self._similarity_cache) < 500:
            self._similarity_cache[cache_key] = similarities
        
        return similarities
    
    def _compute_similarity_matrix(self):
        """Legacy method - now just sets a flag"""
        print(f"Using on-demand {self.similarity_metric} similarity computation for memory efficiency")
        self.similarity_matrix = "on_demand"  # Flag to indicate on-demand computation
    
    def get_restaurant_index(self, restaurant_name):
        """Get restaurant index by name"""
        if restaurant_name not in self.name_to_idx:
            # Fuzzy matching
            restaurant_name_lower = restaurant_name.lower()
            possible_matches = [name for name in self.name_to_idx.keys() 
                              if restaurant_name_lower in name.lower()]
            if possible_matches:
                return self.name_to_idx[possible_matches[0]]
            else:
                raise ValueError(f"Restaurant '{restaurant_name}' not found!")
        return self.name_to_idx[restaurant_name]
    
    def get_recommendations(self, restaurant_name, n_recommendations=10, 
                          include_similar_cuisine=True, include_similar_location=True):
        """
        Get restaurant recommendations with optimized computation
        
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
        
        # Compute similarities on-demand instead of using precomputed matrix
        similarity_scores = self._compute_similarity_for_restaurant(restaurant_idx)
        
        # Apply filters and boosts
        if include_similar_cuisine or include_similar_location:
            similarity_scores = self._apply_filters_and_boosts(
                restaurant_idx, similarity_scores.copy(), 
                include_similar_cuisine, include_similar_location
            )
        
        # Get top similar restaurants (excluding the restaurant itself)
        similarity_scores[restaurant_idx] = -1  # Exclude self
        
        # Use argpartition for efficient top-k selection
        if n_recommendations < len(similarity_scores):
            top_indices = np.argpartition(similarity_scores, -n_recommendations)[-n_recommendations:]
            top_indices = top_indices[np.argsort(similarity_scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(similarity_scores)[::-1][:n_recommendations]
        
        # Create recommendations DataFrame
        recommendations = self.restaurant_data.iloc[top_indices].copy()
        recommendations['similarity_score'] = similarity_scores[top_indices]
        
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
        """Apply cuisine and location boosts to similarity scores (optimized)"""
        reference_restaurant = self.restaurant_data.iloc[restaurant_idx]
        
        if include_similar_cuisine:
            # Optimized cuisine boost with vectorized operations
            reference_cuisines = set(str(reference_restaurant['cuisines']).lower().split(', '))
            
            # Only process restaurants with non-zero similarity scores
            nonzero_indices = np.where(similarity_scores > 0)[0]
            for idx in nonzero_indices:
                if idx != restaurant_idx:
                    restaurant_cuisines = set(str(self.restaurant_data.iloc[idx]['cuisines']).lower().split(', '))
                    cuisine_overlap = len(reference_cuisines.intersection(restaurant_cuisines))
                    if cuisine_overlap > 0:
                        similarity_scores[idx] *= (1 + 0.2 * cuisine_overlap)
        
        if include_similar_location:
            # Vectorized location boost
            reference_location = reference_restaurant['location']
            same_location_mask = (self.restaurant_data['location'] == reference_location).values
            same_location_mask[restaurant_idx] = False  # Exclude self
            similarity_scores[same_location_mask] *= 1.3
        
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
        """Find restaurants based on specific criteria (optimized with vectorized operations)"""
        # Use boolean indexing for efficient filtering
        mask = np.ones(len(self.restaurant_data), dtype=bool)
        
        if cuisine:
            mask &= self.restaurant_data['cuisines'].str.contains(cuisine, case=False, na=False).values
        
        if location:
            mask &= self.restaurant_data['location'].str.contains(location, case=False, na=False).values
        
        if price_range:
            cost_values = self.restaurant_data['cost_for_two'].values
            if price_range == 'budget':
                mask &= (cost_values <= 300)
            elif price_range == 'mid':
                mask &= ((cost_values > 300) & (cost_values <= 600))
            elif price_range == 'expensive':
                mask &= (cost_values > 600)
        
        if rating_min:
            mask &= (self.restaurant_data['rating'].values >= rating_min)
        
        # Apply filter and sort efficiently
        filtered_data = self.restaurant_data[mask]
        
        # Sort by rating and popularity
        filtered_data = filtered_data.sort_values(['rating', 'votes'], ascending=[False, False])
        
        return filtered_data.head(n_results)