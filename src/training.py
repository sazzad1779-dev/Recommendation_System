# ========================================
# FIXED MODEL TRAINING (03_model_training.ipynb)
# ========================================

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings('ignore')

# ========================================
# STEP 1: LOAD PROCESSED DATA WITH VALIDATION
# ========================================

def load_and_validate_data():
    """Load processed data with comprehensive validation"""
    
    print("=== LOADING PROCESSED DATA ===")
    
    try:
        # Load main datasets
        restaurant_data = pd.read_csv('processed_restaurant_data.csv')
        content_features = pd.read_csv('content_features.csv')
        text_features = pd.read_csv('text_features.csv')
        hybrid_features = pd.read_csv('hybrid_features.csv')
        
        print(f"Restaurant data shape: {restaurant_data.shape}")
        print(f"Content features shape: {content_features.shape}")
        print(f"Text features shape: {text_features.shape}")
        print(f"Hybrid features shape: {hybrid_features.shape}")
        
        # Validate that all datasets have the same number of rows
        shapes = [restaurant_data.shape[0], content_features.shape[0], 
                 text_features.shape[0], hybrid_features.shape[0]]
        
        if len(set(shapes)) != 1:
            raise ValueError(f"Shape mismatch! Shapes: {shapes}")
        
        # Check for NaN values
        print(f"\n=== NaN VALIDATION ===")
        print(f"NaNs in restaurant_data: {restaurant_data.isna().sum().sum()}")
        print(f"NaNs in content_features: {content_features.isna().sum().sum()}")
        print(f"NaNs in text_features: {text_features.isna().sum().sum()}")
        print(f"NaNs in hybrid_features: {hybrid_features.isna().sum().sum()}")
        
        # Final NaN cleanup if any remain
        content_features = content_features.fillna(0)
        text_features = text_features.fillna(0)
        hybrid_features = hybrid_features.fillna(0)
        
        # Ensure restaurant_data has no NaNs in critical columns
        restaurant_data['rating'] = restaurant_data['rating'].fillna(restaurant_data['rating'].median())
        restaurant_data['cost_for_two'] = restaurant_data['cost_for_two'].fillna(restaurant_data['cost_for_two'].median())
        restaurant_data = restaurant_data.fillna('Unknown')
        
        print(f"After cleanup - NaNs in content_features: {content_features.isna().sum().sum()}")
        print(f"After cleanup - NaNs in text_features: {text_features.isna().sum().sum()}")
        print(f"After cleanup - NaNs in hybrid_features: {hybrid_features.isna().sum().sum()}")
        
        # Check for infinite values
        print(f"\n=== INFINITE VALUES CHECK ===")
        # print(f"Inf in content_features: {np.isinf(content_features.values).sum()}")
        # print(f"Inf in text_features: {np.isinf(text_features.values).sum()}")
        # print(f"Inf in hybrid_features: {np.isinf(hybrid_features.values).sum()}")
        
        # Replace infinite values with large finite values
        content_features = content_features.replace([np.inf, -np.inf], [1e6, -1e6])
        text_features = text_features.replace([np.inf, -np.inf], [1e6, -1e6])
        hybrid_features = hybrid_features.replace([np.inf, -np.inf], [1e6, -1e6])
        
        print("Data loading and validation complete!")
        return restaurant_data, content_features, text_features, hybrid_features
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

# Load and validate data
restaurant_data, content_features, text_features, hybrid_features = load_and_validate_data()

if restaurant_data is None:
    raise ValueError("Failed to load data. Please run feature engineering first.")

# ========================================
# STEP 2: ROBUST RECOMMENDER BASE CLASS
# ========================================

class RobustRestaurantRecommender:
    def __init__(self, restaurant_data, feature_matrix, similarity_metric='cosine'):
        """
        Initialize robust recommender with NaN handling
        
        Args:
            restaurant_data: DataFrame with restaurant information
            feature_matrix: DataFrame with features for similarity calculation
            similarity_metric: 'cosine', 'euclidean', or 'knn'
        """
        self.restaurant_data = restaurant_data.reset_index(drop=True)
        self.feature_matrix = feature_matrix.reset_index(drop=True)
        self.similarity_metric = similarity_metric
        
        # Validate inputs
        self._validate_inputs()
        
        # Create restaurant name to index mapping
        self.name_to_idx = {name: idx for idx, name in enumerate(self.restaurant_data['name'])}
        
        # Precompute similarity matrix
        self._compute_similarity_matrix()
        
    def _validate_inputs(self):
        """Validate input data"""
        
        # Check shapes match
        if len(self.restaurant_data) != len(self.feature_matrix):
            raise ValueError(f"Shape mismatch: restaurant_data {len(self.restaurant_data)} vs feature_matrix {len(self.feature_matrix)}")
        
        # Check for NaN or infinite values
        if self.feature_matrix.isna().any().any():
            print("WARNING: NaN values detected in feature matrix. Filling with 0.")
            self.feature_matrix = self.feature_matrix.fillna(0)
            
        if np.isinf(self.feature_matrix.values).any():
            print("WARNING: Infinite values detected in feature matrix. Replacing with finite values.")
            self.feature_matrix = self.feature_matrix.replace([np.inf, -np.inf], [1e6, -1e6])
        
        # Ensure numeric types
        self.feature_matrix = self.feature_matrix.astype(float)
        
        print(f"Validation complete. Feature matrix shape: {self.feature_matrix.shape}")
        
    def _compute_similarity_matrix(self):
        """Compute similarity matrix with error handling"""
        
        print(f"Computing {self.similarity_metric} similarity matrix...")
        
        try:
            if self.similarity_metric == 'cosine':
                # Handle potential division by zero in cosine similarity
                from sklearn.preprocessing import normalize
                normalized_features = normalize(self.feature_matrix, norm='l2')
                self.similarity_matrix = cosine_similarity(normalized_features)
                
            elif self.similarity_metric == 'euclidean':
                distances = euclidean_distances(self.feature_matrix)
                # Convert distances to similarities (higher = more similar)
                max_distance = distances.max()
                if max_distance == 0:
                    max_distance = 1  # Prevent division by zero
                self.similarity_matrix = 1 - (distances / max_distance)
                
            elif self.similarity_metric == 'knn':
                # Use KNN for similarity
                self.knn_model = NearestNeighbors(
                    n_neighbors=min(50, len(self.feature_matrix)), 
                    metric='cosine', 
                    algorithm='brute'
                )
                self.knn_model.fit(self.feature_matrix)
                self.similarity_matrix = None  # Will compute on-demand
            
            print(f"Similarity computation complete!")
            
        except Exception as e:
            print(f"Error computing similarity matrix: {e}")
            # Fallback to simple dot product similarity
            normalized_features = self.feature_matrix / (np.linalg.norm(self.feature_matrix, axis=1, keepdims=True) + 1e-8)
            self.similarity_matrix = np.dot(normalized_features, normalized_features.T)
            print("Using fallback dot product similarity.")
    
    def get_restaurant_index(self, restaurant_name):
        """Get restaurant index by name with fuzzy matching"""
        
        if restaurant_name in self.name_to_idx:
            return self.name_to_idx[restaurant_name]
        
        # Fuzzy matching
        restaurant_name_lower = restaurant_name.lower()
        possible_matches = []
        
        for name, idx in self.name_to_idx.items():
            if restaurant_name_lower in name.lower() or name.lower() in restaurant_name_lower:
                possible_matches.append((name, idx))
        
        if possible_matches:
            # Return the first match
            return possible_matches[0][1]
        else:
            # If no match found, suggest similar names
            similar_names = [name for name in self.name_to_idx.keys() 
                           if any(word in name.lower() for word in restaurant_name_lower.split())][:5]
            raise ValueError(f"Restaurant '{restaurant_name}' not found! Similar names: {similar_names}")
    
    def get_recommendations(self, restaurant_name, n_recommendations=10, 
                          include_filters=True):
        """
        Get restaurant recommendations with error handling
        
        Args:
            restaurant_name: Name of the reference restaurant
            n_recommendations: Number of recommendations to return
            include_filters: Whether to apply cuisine/location filters
        """
        try:
            restaurant_idx = self.get_restaurant_index(restaurant_name)
        except ValueError as e:
            return str(e)
        
        try:
            if self.similarity_metric == 'knn':
                # Use KNN for recommendations
                n_neighbors = min(n_recommendations + 1, len(self.feature_matrix))
                distances, indices = self.knn_model.kneighbors(
                    [self.feature_matrix.iloc[restaurant_idx]], 
                    n_neighbors=n_neighbors
                )
                # Remove the restaurant itself (first result)
                recommended_indices = indices[0][1:]
                similarity_scores = 1 - distances[0][1:]  # Convert distances to similarities
            else:
                # Use precomputed similarity matrix
                similarity_scores = self.similarity_matrix[restaurant_idx].copy()
                
                # Apply filters if requested
                if include_filters:
                    similarity_scores = self._apply_smart_filters(restaurant_idx, similarity_scores)
                
                # Get top similar restaurants (excluding the restaurant itself)
                similarity_scores[restaurant_idx] = -1  # Ensure self is not recommended
                recommended_indices = np.argsort(similarity_scores)[::-1][:n_recommendations]
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
            
        except Exception as e:
            return f"Error generating recommendations: {str(e)}"
    
    def _apply_smart_filters(self, restaurant_idx, similarity_scores):
        """Apply intelligent filters to boost relevant recommendations"""
        
        try:
            reference_restaurant = self.restaurant_data.iloc[restaurant_idx]
            
            # Boost restaurants with similar cuisines
            reference_cuisines = str(reference_restaurant['cuisines']).lower().split(',')
            reference_cuisines = [c.strip() for c in reference_cuisines if c.strip()]
            
            # Boost same location restaurants
            reference_location = str(reference_restaurant['location'])
            
            # Boost similar price range restaurants
            reference_cost = reference_restaurant.get('cost_for_two', 0)
            
            for idx in range(len(similarity_scores)):
                if idx != restaurant_idx:
                    try:
                        current_restaurant = self.restaurant_data.iloc[idx]
                        boost_factor = 1.0
                        
                        # Cuisine similarity boost
                        current_cuisines = str(current_restaurant['cuisines']).lower().split(',')
                        current_cuisines = [c.strip() for c in current_cuisines if c.strip()]
                        
                        cuisine_overlap = len(set(reference_cuisines) & set(current_cuisines))
                        if cuisine_overlap > 0:
                            boost_factor += 0.15 * cuisine_overlap
                        
                        # Location boost
                        if str(current_restaurant['location']) == reference_location:
                            boost_factor += 0.25
                        
                        # Price range boost
                        current_cost = current_restaurant.get('cost_for_two', 0)
                        if abs(current_cost - reference_cost) < 200:  # Within ₹200
                            boost_factor += 0.10
                        
                        similarity_scores[idx] *= boost_factor
                        
                    except Exception:
                        continue  # Skip problematic restaurants
            
            return similarity_scores
            
        except Exception as e:
            print(f"Filter application failed: {e}")
            return similarity_scores
    
    def get_restaurant_details(self, restaurant_name):
        """Get detailed information about a restaurant"""
        try:
            restaurant_idx = self.get_restaurant_index(restaurant_name)
            return self.restaurant_data.iloc[restaurant_idx].to_dict()
        except ValueError as e:
            return {"error": str(e)}
    
    def find_restaurants_by_criteria(self, cuisine=None, location=None, 
                                   price_range=None, rating_min=None, n_results=20):
        """Find restaurants based on specific criteria with error handling"""
        
        try:
            filtered_data = self.restaurant_data.copy()
            
            # Apply filters with error handling
            if cuisine:
                mask = filtered_data['cuisines'].str.contains(cuisine, case=False, na=False)
                filtered_data = filtered_data[mask]
            
            if location:
                mask = filtered_data['location'].str.contains(location, case=False, na=False)
                filtered_data = filtered_data[mask]
            
            if price_range:
                cost_col = 'cost_for_two'
                if price_range.lower() == 'budget':
                    filtered_data = filtered_data[filtered_data[cost_col] <= 300]
                elif price_range.lower() in ['mid', 'medium']:
                    filtered_data = filtered_data[
                        (filtered_data[cost_col] > 300) & (filtered_data[cost_col] <= 600)
                    ]
                elif price_range.lower() in ['expensive', 'high']:
                    filtered_data = filtered_data[filtered_data[cost_col] > 600]
            
            if rating_min:
                filtered_data = filtered_data[filtered_data['rating'] >= rating_min]
            
            # Sort by rating and return top results
            filtered_data = filtered_data.sort_values(['rating', 'name'], ascending=[False, True])
            
            return filtered_data.head(n_results)
            
        except Exception as e:
            return f"Error in criteria search: {str(e)}"


# ========================================
# STEP 3: SPECIALIZED RECOMMENDER CLASSES
# ========================================

class ContentBasedRecommender(RobustRestaurantRecommender):
    """Content-based recommender focusing on restaurant attributes"""
    
    def __init__(self, restaurant_data, feature_matrix):
        super().__init__(restaurant_data, feature_matrix, similarity_metric='cosine')
        self.recommender_type = 'content_based'
    
    def get_cuisine_recommendations(self, restaurant_name, n_recommendations=10):
        """Get recommendations based primarily on cuisine similarity"""
        return self.get_recommendations(restaurant_name, n_recommendations, include_filters=True)
    
    def get_location_recommendations(self, restaurant_name, n_recommendations=10):
        """Get recommendations based on location and nearby restaurants"""
        return self.get_recommendations(restaurant_name, n_recommendations, include_filters=True)


class TextBasedRecommender(RobustRestaurantRecommender):
    """Text-based recommender using reviews and descriptions"""
    
    def __init__(self, restaurant_data, feature_matrix):
        super().__init__(restaurant_data, feature_matrix, similarity_metric='cosine')
        self.recommender_type = 'text_based'


class HybridRecommender:
    """Advanced hybrid recommender combining multiple approaches"""
    
    def __init__(self, restaurant_data, content_features, text_features):
        self.restaurant_data = restaurant_data
        
        
        # Initialize individual recommenders with error handling
        try:
            self.content_recommender = ContentBasedRecommender(restaurant_data, content_features)
            print("Content-based recommender initialized successfully")
        except Exception as e:
            print(f"Error initializing content recommender: {e}")
            self.content_recommender = None
        
        try:
            self.text_recommender = TextBasedRecommender(restaurant_data, text_features)
            print("Text-based recommender initialized successfully")
        except Exception as e:
            print(f"Error initializing text recommender: {e}")
            self.text_recommender = None
        
        # Weights for different recommendation types
        self.weights = {
            'content': 0.65,
            'text': 0.35
        }
        
        self.recommender_type = 'hybrid'
    
    def get_hybrid_recommendations(self, restaurant_name, n_recommendations=10):
        """Get recommendations using hybrid approach with error handling"""
        
        try:
            # Get recommendations from available recommenders
            content_recs = None
            text_recs = None
            
            if self.content_recommender:
                content_recs = self.content_recommender.get_recommendations(
                    restaurant_name, n_recommendations * 2
                )
            
            if self.text_recommender:
                text_recs = self.text_recommender.get_recommendations(
                    restaurant_name, n_recommendations * 2
                )
            
            # Handle errors from individual recommenders
            if isinstance(content_recs, str):
                content_recs = None
            if isinstance(text_recs, str):
                text_recs = None
            
            if content_recs is None and text_recs is None:
                return "No recommendations available from any method"
            
            # If only one method available, use it
            if content_recs is None:
                return text_recs.head(n_recommendations)
            if text_recs is None:
                return content_recs.head(n_recommendations)
            
            # Combine recommendations with weighted scoring
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
                try:
                    restaurant_info = self.restaurant_data[self.restaurant_data['name'] == name].iloc[0]
                    final_recommendations.append({
                        'name': restaurant_info['name'],
                        'cuisines': restaurant_info['cuisines'],
                        'location': restaurant_info['location'],
                        'rating': restaurant_info['rating'],
                        'cost_for_two': restaurant_info['cost_for_two'],
                        'rest_type': restaurant_info.get('rest_type', 'Unknown'),
                        'online_order': restaurant_info.get('online_order', 'Unknown'),
                        'book_table': restaurant_info.get('book_table', 'Unknown'),
                        'hybrid_score': combined_scores[name]
                    })
                except Exception:
                    continue  # Skip restaurants with missing data
            
            return pd.DataFrame(final_recommendations)
            
        except Exception as e:
            return f"Error in hybrid recommendations: {str(e)}"
    
    def update_weights(self, content_weight, text_weight):
        """Update recommendation weights"""
        total = content_weight + text_weight
        self.weights = {
            'content': content_weight / total,
            'text': text_weight / total
        }
        print(f"Updated weights: Content={self.weights['content']:.2f}, Text={self.weights['text']:.2f}")


# ========================================
# STEP 4: INITIALIZE RECOMMENDERS WITH ERROR HANDLING
# ========================================

def initialize_recommenders():
    """Initialize all recommenders with comprehensive error handling"""
    
    print("=== INITIALIZING RECOMMENDERS ===")
    
    recommenders = {}
    
    # Initialize content-based recommender
    try:
        content_recommender = ContentBasedRecommender(restaurant_data, content_features)
        recommenders['content'] = content_recommender
        print("✓ Content-based recommender initialized")
    except Exception as e:
        print(f"✗ Content-based recommender failed: {e}")
        recommenders['content'] = None
    
    # Initialize text-based recommender  
    try:
        text_recommender = TextBasedRecommender(restaurant_data, text_features)
        recommenders['text'] = text_recommender
        print("✓ Text-based recommender initialized")
    except Exception as e:
        print(f"✗ Text-based recommender failed: {e}")
        recommenders['text'] = None
    
    # Initialize hybrid recommender
    try:
        hybrid_recommender = HybridRecommender(restaurant_data, content_features, text_features)
        recommenders['hybrid'] = hybrid_recommender
        print("✓ Hybrid recommender initialized")
    except Exception as e:
        print(f"✗ Hybrid recommender failed: {e}")
        recommenders['hybrid'] = None
    
    # Check if at least one recommender works
    working_recommenders = [k for k, v in recommenders.items() if v is not None]
    print(f"\nWorking recommenders: {working_recommenders}")
    
    if not working_recommenders:
        raise ValueError("No recommenders could be initialized!")
    
    return recommenders

# Initialize recommenders
recommenders = initialize_recommenders()

# ========================================
# STEP 5: TESTING AND VALIDATION
# ========================================

def test_recommenders():
    """Test all recommenders with sample data"""
    
    print("=== TESTING RECOMMENDERS ===")
    
    # Get a sample restaurant for testing
    sample_restaurant = restaurant_data['name'].iloc[0]
    print(f"Testing with restaurant: '{sample_restaurant}'")
    
    # Test each recommender
    for rec_type, recommender in recommenders.items():
        if recommender is None:
            print(f"\n{rec_type.upper()}: SKIPPED (not initialized)")
            continue
            
        print(f"\n{rec_type.upper()} RECOMMENDATIONS:")
        try:
            if rec_type == 'hybrid':
                recs = recommender.get_hybrid_recommendations(sample_restaurant, 5)
                if isinstance(recs, pd.DataFrame) and not recs.empty:
                    print(recs[['name', 'cuisines', 'location', 'hybrid_score']].to_string(index=False))
                else:
                    print(f"Result: {recs}")
            else:
                recs = recommender.get_recommendations(sample_restaurant, 5)
                if isinstance(recs, pd.DataFrame) and not recs.empty:
                    print(recs[['name', 'cuisines', 'location', 'similarity_score']].to_string(index=False))
                else:
                    print(f"Result: {recs}")
                    
        except Exception as e:
            print(f"Error testing {rec_type}: {e}")

# Run tests
test_recommenders()

# ========================================
# STEP 6: INTERACTIVE FUNCTIONS
# ========================================

def get_recommendation_interactive():
    """Interactive function to get recommendations"""
    
    print("\n=== INTERACTIVE RECOMMENDATION SYSTEM ===")
    print("Available recommenders:", [k for k, v in recommenders.items() if v is not None])
    
    # Get user input
    restaurant_name = input("\nEnter restaurant name: ").strip()
    if not restaurant_name:
        print("Please enter a restaurant name.")
        return
    
    method = input("Choose method (content/text/hybrid): ").strip().lower()
    if method not in recommenders or recommenders[method] is None:
        print(f"Method '{method}' not available. Using hybrid.")
        method = 'hybrid'
    
    try:
        n_recs = int(input("Number of recommendations (default 10): ") or "10")
    except:
        n_recs = 10
    
    print(f"\nGetting {method} recommendations for '{restaurant_name}'...")
    
    # Get recommendations
    try:
        recommender = recommenders[method]
        if method == 'hybrid':
            results = recommender.get_hybrid_recommendations(restaurant_name, n_recs)
        else:
            results = recommender.get_recommendations(restaurant_name, n_recs)
        
        if isinstance(results, str):
            print(f"Error: {results}")
        elif isinstance(results, pd.DataFrame) and not results.empty:
            print(f"\nRecommendations:")
            print(results.to_string(index=False))
        else:
            print("No recommendations found.")
            
    except Exception as e:
        print(f"Error: {e}")

def search_by_criteria_interactive():
    """Interactive search by criteria"""
    
    print("\n=== SEARCH BY CRITERIA ===")
    
    cuisine = input("Enter cuisine (optional): ").strip() or None
    location = input("Enter location (optional): ").strip() or None  
    price_range = input("Enter price range - budget/mid/expensive (optional): ").strip() or None
    try:
        rating_min = float(input("Enter minimum rating (optional): ") or "0")
    except:
        rating_min = None
    
    # Use content recommender for criteria search
    content_rec = recommenders.get('content')
    if content_rec is None:
        print("Content-based recommender not available for criteria search.")
        return
    
    try:
        results = content_rec.find_restaurants_by_criteria(
            cuisine=cuisine, location=location, 
            price_range=price_range, rating_min=rating_min
        )
        
        if isinstance(results, str):
            print(f"Error: {results}")
        elif isinstance(results, pd.DataFrame) and not results.empty:
            print(f"\nFound {len(results)} restaurants:")
            cols_to_show = ['name', 'cuisines', 'location', 'rating', 'cost_for_two']
            print(results[cols_to_show].to_string(index=False))
        else:
            print("No restaurants found matching criteria.")
            
    except Exception as e:
        print(f"Error: {e}")

# ========================================
# STEP 7: SAVE MODELS
# ========================================

def save_models():
    """Save all working models"""
    
    print("=== SAVING MODELS ===")
    
    saved_models = []
    
    for rec_type, recommender in recommenders.items():
        if recommender is not None:
            try:
                filename = f'{rec_type}_recommender.pkl'
                with open(filename, 'wb') as f:
                    pickle.dump(recommender, f)
                saved_models.append(filename)
                print(f"✓ Saved {filename}")
            except Exception as e:
                print(f"✗ Failed to save {rec_type}: {e}")
    
    print(f"\nSuccessfully saved {len(saved_models)} models: {saved_models}")
    return saved_models

# Save models
saved_files = save_models()


# ========================================
# STEP 8: API-READY FUNCTIONS
# ========================================

def api_get_recommendations(restaurant_name, method='hybrid', n_recommendations=10):
    """API-friendly recommendation function"""
    
    try:
        if method not in recommenders or recommenders[method] is None:
            available_methods = [k for k, v in recommenders.items() if v is not None]
            return {
                'status': 'error',
                'message': f'Method {method} not available. Available methods: {available_methods}'
            }
        
        recommender = recommenders[method]
        
        if method == 'hybrid':
            recs = recommender.get_hybrid_recommendations(restaurant_name, n_recommendations)
            score_col = 'hybrid_score'
        else:
            recs = recommender.get_recommendations(restaurant_name, n_recommendations)
            score_col = 'similarity_score'
        
        if isinstance(recs, str):
            return {'status': 'error', 'message': recs}
        
        if isinstance(recs, pd.DataFrame) and not recs.empty:
            recommendations = []
            for _, row in recs.iterrows():
                rec = {
                    'name': row['name'],
                    'cuisines': row['cuisines'],
                    'location': row['location'],
                    'rating': float(row['rating']) if pd.notna(row['rating']) else None,
                    'cost_for_two': float(row['cost_for_two']) if pd.notna(row['cost_for_two']) else None,
                    'restaurant_type': row.get('rest_type', 'Unknown'),
                    'online_order': row.get('online_order', 'Unknown'),
                    'table_booking': row.get('book_table', 'Unknown'),
                    'score': float(row[score_col])
                }
                recommendations.append(rec)
            
            return {
                'status': 'success',
                'input_restaurant': restaurant_name,
                'method_used': method,
                'count': len(recommendations),
                'recommendations': recommendations
            }
        else:
            return {
                'status': 'success',
                'input_restaurant': restaurant_name,
                'method_used': method,
                'count': 0,
                'recommendations': []
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'message': f'Unexpected error: {str(e)}'
        }

# ========================================
# FINAL SUMMARY
# ========================================

print("\n" + "="*60)
print("MODEL TRAINING COMPLETE!")
print("="*60)

print(f"✓ Restaurant data loaded: {len(restaurant_data)} restaurants")
print(f"✓ Feature matrices prepared with no NaN values")
print(f"✓ Recommenders initialized: {list(k for k, v in recommenders.items() if v is not None)}")
print(f"✓ Models saved: {saved_files}")

print(f"\n📊 DATASET SUMMARY:")
print(f"   • Total restaurants: {len(restaurant_data)}")
print(f"   • Content features: {content_features.shape[1]} dimensions")  
print(f"   • Text features: {text_features.shape[1]} dimensions")
print(f"   • Hybrid features: {hybrid_features.shape[1]} dimensions")

print(f"\n🎯 READY TO USE:")
print("   • get_recommendation_interactive() - Interactive recommendations")
print("   • search_by_criteria_interactive() - Search by filters") 
print("   • api_get_recommendations() - API function")

print(f"\n🚀 All systems ready for evaluation phase!")