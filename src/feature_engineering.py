# ========================================
# FIXED FEATURE ENGINEERING (02_feature_engineering.ipynb)
# ========================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.impute import SimpleImputer
import re
import ast
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('/Users/jbc/Documents/sazzad/Recommendation_System/data/zomato.csv').sample(frac=0.1, random_state=42)
print(f"Original dataset shape: {df.shape}")

# ========================================
# STEP 1: COMPREHENSIVE DATA CLEANING
# ========================================

def comprehensive_data_cleaning(df):
    """Comprehensive data cleaning with proper NaN handling"""
    
    df_clean = df.copy()
    
    # 1. Clean rating column
    def clean_rating(rate):
        if pd.isna(rate) or str(rate).strip() in ['NEW', '-', 'nan', '']:
            return np.nan
        try:
            # Extract numeric part before '/'
            rate_str = str(rate).strip()
            if '/' in rate_str:
                return float(rate_str.split('/')[0])
            else:
                # Handle cases like '4.1' without '/'
                return float(rate_str)
        except:
            return np.nan

    df_clean['rating'] = df_clean['rate'].apply(clean_rating)
    
    # 2. Clean cost column
    def clean_cost(cost):
        if pd.isna(cost):
            return np.nan
        try:
            # Remove commas, currency symbols, and extra whitespace
            cost_str = str(cost).replace(',', '').replace('₹', '').replace('Rs.', '').strip()
            if cost_str == '' or cost_str == 'nan':
                return np.nan
            return float(cost_str)
        except:
            return np.nan

    df_clean['cost_for_two'] = df_clean['approx_cost(for two people)'].apply(clean_cost)
    
    # 3. Clean text fields
    df_clean['location'] = df_clean['location'].fillna('Unknown Location')
    df_clean['rest_type'] = df_clean['rest_type'].fillna('Not Specified')
    df_clean['cuisines'] = df_clean['cuisines'].fillna('Not Specified')
    df_clean['dish_liked'] = df_clean['dish_liked'].fillna('')
    df_clean['phone'] = df_clean['phone'].str.replace(r'\r\n.*', '', regex=True).fillna('Not Available')
    
    # 4. Handle votes (should not have NaN)
    df_clean['votes'] = df_clean['votes'].fillna(0).astype(int)
    
    # 5. Clean binary fields
    df_clean['online_order'] = df_clean['online_order'].fillna('No')
    df_clean['book_table'] = df_clean['book_table'].fillna('No')
    
    # 6. Clean categorical fields
    df_clean['listed_in(type)'] = df_clean['listed_in(type)'].fillna('Other')
    df_clean['listed_in(city)'] = df_clean['listed_in(city)'].fillna('Other')
    
    print("=== DATA CLEANING SUMMARY ===")
    print(f"Original NaN in rating: {df['rate'].isna().sum()}")
    print(f"Final NaN in rating: {df_clean['rating'].isna().sum()}")
    print(f"Original NaN in cost: {df['approx_cost(for two people)'].isna().sum()}")
    print(f"Final NaN in cost: {df_clean['cost_for_two'].isna().sum()}")
    
    return df_clean

# Apply comprehensive cleaning
df_clean = comprehensive_data_cleaning(df)

# ========================================
# STEP 2: ADVANCED TEXT PROCESSING
# ========================================

def advanced_text_processing(df_clean):
    """Advanced text processing with proper NaN handling"""
    
    # 1. Process cuisines
    def process_cuisines(cuisines):
        if pd.isna(cuisines) or cuisines == 'Not Specified':
            return []
        try:
            return [cuisine.strip().lower() for cuisine in str(cuisines).split(',')]
        except:
            return []

    df_clean['cuisine_list'] = df_clean['cuisines'].apply(process_cuisines)
    
    # 2. Process dishes liked
    def process_dishes(dishes):
        if pd.isna(dishes) or dishes == '':
            return []
        try:
            # Handle various separators
            dishes_str = str(dishes).replace(';', ',')
            return [dish.strip().lower() for dish in dishes_str.split(',') if dish.strip()]
        except:
            return []

    df_clean['dish_list'] = df_clean['dish_liked'].apply(process_dishes)
    
    # 3. Create combined text features
    def create_combined_features(row):
        """Create combined text features for similarity"""
        features = []
        
        # Add cuisines
        if row['cuisines'] and row['cuisines'] != 'Not Specified':
            features.append(str(row['cuisines']))
        
        # Add dishes
        if row['dish_liked'] and row['dish_liked'] != '':
            features.append(str(row['dish_liked']))
        
        # Add restaurant type
        if row['rest_type'] and row['rest_type'] != 'Not Specified':
            features.append(str(row['rest_type']))
        
        # Add location for context
        if row['location'] and row['location'] != 'Unknown Location':
            features.append(str(row['location']))
        
        return ' '.join(features) if features else 'no_features'

    df_clean['combined_features'] = df_clean.apply(create_combined_features, axis=1)
    
    # 4. Process reviews
    def extract_review_text(reviews_list):
        if pd.isna(reviews_list) or reviews_list == '[]' or reviews_list == '':
            return 'no_review'
        try:
            reviews = ast.literal_eval(str(reviews_list))
            review_texts = []
            for review in reviews:
                if isinstance(review, tuple) and len(review) > 1:
                    review_texts.append(str(review[1]))
                elif isinstance(review, str):
                    review_texts.append(review)
            return ' '.join(review_texts) if review_texts else 'no_review'
        except:
            return 'no_review'

    df_clean['review_text'] = df_clean['reviews_list'].apply(extract_review_text)
    
    print("=== TEXT PROCESSING COMPLETE ===")
    print(f"Restaurants with cuisines: {(df_clean['cuisine_list'].apply(len) > 0).sum()}")
    print(f"Restaurants with dishes: {(df_clean['dish_list'].apply(len) > 0).sum()}")
    print(f"Combined features sample: {df_clean['combined_features'].iloc[0]}")
    
    return df_clean

# Apply text processing
df_clean = advanced_text_processing(df_clean)

# ========================================
# STEP 3: NUMERICAL FEATURE ENGINEERING WITH IMPUTATION
# ========================================

def create_numerical_features(df_clean):
    """Create and impute numerical features"""
    
    # 1. Handle missing numerical values with proper imputation
    # Create imputers
    rating_imputer = SimpleImputer(strategy='median')
    cost_imputer = SimpleImputer(strategy='median')
    
    # Fit and transform
    df_clean['rating_imputed'] = rating_imputer.fit_transform(df_clean[['rating']]).flatten()
    df_clean['cost_imputed'] = cost_imputer.fit_transform(df_clean[['cost_for_two']]).flatten()
    
    # 2. Create derived features
    df_clean['log_votes'] = np.log1p(df_clean['votes'])  # Log transform votes
    df_clean['rating_votes_interaction'] = df_clean['rating_imputed'] * df_clean['log_votes']
    df_clean['cost_per_rating'] = df_clean['cost_imputed'] / (df_clean['rating_imputed'] + 0.1)
    df_clean['popularity_score'] = df_clean['rating_imputed'] * df_clean['log_votes']
    
    # 3. Create categorical numerical features
    # Binary features
    df_clean['online_order_binary'] = (df_clean['online_order'] == 'Yes').astype(int)
    df_clean['book_table_binary'] = (df_clean['book_table'] == 'Yes').astype(int)
    
    # Price categories
    def categorize_price(cost):
        if cost <= 300:
            return 0  # Budget
        elif cost <= 600:
            return 1  # Mid-range
        elif cost <= 1200:
            return 2  # Expensive
        else:
            return 3  # Premium

    df_clean['price_category'] = df_clean['cost_imputed'].apply(categorize_price)
    
    # Rating categories
    def categorize_rating(rating):
        if rating < 3.0:
            return 0  # Poor
        elif rating < 3.5:
            return 1  # Average
        elif rating < 4.0:
            return 2  # Good
        else:
            return 3  # Excellent

    df_clean['rating_category'] = df_clean['rating_imputed'].apply(categorize_rating)
    
    # 4. Scale numerical features
    scaler = StandardScaler()
    numerical_columns = ['rating_imputed', 'cost_imputed', 'log_votes', 
                        'rating_votes_interaction', 'popularity_score']
    
    scaled_features = scaler.fit_transform(df_clean[numerical_columns])
    scaled_df = pd.DataFrame(scaled_features, 
                           columns=[col + '_scaled' for col in numerical_columns],
                           index=df_clean.index)
    
    # Add scaled features to main dataframe
    df_clean = pd.concat([df_clean, scaled_df], axis=1)
    
    print("=== NUMERICAL FEATURES COMPLETE ===")
    print(f"No NaN in rating_imputed: {df_clean['rating_imputed'].isna().sum() == 0}")
    print(f"No NaN in cost_imputed: {df_clean['cost_imputed'].isna().sum() == 0}")
    print(f"Numerical features shape: {scaled_df.shape}")
    
    return df_clean, scaler

# Create numerical features
df_clean, feature_scaler = create_numerical_features(df_clean)

# ========================================
# STEP 4: CATEGORICAL ENCODING WITH NO NaNs
# ========================================

def create_categorical_features(df_clean):
    """Create categorical features with proper encoding"""
    
    # 1. Location encoding (group less frequent locations)
    location_counts = df_clean['location'].value_counts()
    popular_locations = location_counts[location_counts >= 20].index  # Minimum 20 restaurants
    
    def group_location(location):
        return location if location in popular_locations else 'Other_Location'
    
    df_clean['location_grouped'] = df_clean['location'].apply(group_location)
    location_encoded = pd.get_dummies(df_clean['location_grouped'], prefix='location')
    
    # 2. Restaurant type encoding
    rest_type_counts = df_clean['rest_type'].value_counts()
    popular_rest_types = rest_type_counts[rest_type_counts >= 10].index
    
    def group_rest_type(rest_type):
        return rest_type if rest_type in popular_rest_types else 'Other_Type'
    
    df_clean['rest_type_grouped'] = df_clean['rest_type'].apply(group_rest_type)
    rest_type_encoded = pd.get_dummies(df_clean['rest_type_grouped'], prefix='rest_type')
    
    # 3. Cuisine encoding using MultiLabelBinarizer
    mlb_cuisine = MultiLabelBinarizer()
    
    # Filter out empty lists and ensure we have at least one cuisine per restaurant
    cuisine_lists_filtered = []
    for cuisine_list in df_clean['cuisine_list']:
        if len(cuisine_list) > 0:
            cuisine_lists_filtered.append(cuisine_list)
        else:
            cuisine_lists_filtered.append(['unknown'])  # Default cuisine for empty lists
    
    cuisine_encoded = mlb_cuisine.fit_transform(cuisine_lists_filtered)
    cuisine_df = pd.DataFrame(cuisine_encoded, 
                             columns=['cuisine_' + col for col in mlb_cuisine.classes_],
                             index=df_clean.index)
    
    # 4. Service features
    service_features = df_clean[['online_order_binary', 'book_table_binary', 
                                'price_category', 'rating_category']].copy()
    
    print("=== CATEGORICAL ENCODING COMPLETE ===")
    print(f"Location features: {location_encoded.shape[1]}")
    print(f"Restaurant type features: {rest_type_encoded.shape[1]}")
    print(f"Cuisine features: {cuisine_df.shape[1]}")
    print(f"Service features: {service_features.shape[1]}")
    
    # Check for NaNs
    print(f"NaNs in location_encoded: {location_encoded.isna().sum().sum()}")
    print(f"NaNs in rest_type_encoded: {rest_type_encoded.isna().sum().sum()}")
    print(f"NaNs in cuisine_df: {cuisine_df.isna().sum().sum()}")
    print(f"NaNs in service_features: {service_features.isna().sum().sum()}")
    
    return location_encoded, rest_type_encoded, cuisine_df, service_features, mlb_cuisine

# Create categorical features
location_encoded, rest_type_encoded, cuisine_df, service_features, cuisine_encoder = create_categorical_features(df_clean)

# ========================================
# STEP 5: TEXT VECTORIZATION WITH NO NaNs
# ========================================

def create_text_features(df_clean):
    """Create TF-IDF features from text data"""
    
    # 1. Combined features TF-IDF
    tfidf_combined = TfidfVectorizer(
        max_features=50,  # Reduced to avoid memory issues
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    # Ensure no NaN values in text
    combined_text = df_clean['combined_features'].fillna('no_features')
    tfidf_matrix = tfidf_combined.fit_transform(combined_text)
    
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=['tfidf_combined_' + str(i) for i in range(tfidf_matrix.shape[1])],
        index=df_clean.index
    )
    
    # 2. Review text TF-IDF (separate)
    tfidf_reviews = TfidfVectorizer(
        max_features=30,  # Smaller for reviews
        stop_words='english',
        min_df=3,
        max_df=0.9
    )
    
    review_text = df_clean['review_text'].fillna('no_review')
    review_matrix = tfidf_reviews.fit_transform(review_text)
    
    review_df = pd.DataFrame(
        review_matrix.toarray(),
        columns=['tfidf_review_' + str(i) for i in range(review_matrix.shape[1])],
        index=df_clean.index
    )
    
    # Combine text features
    text_features_combined = pd.concat([tfidf_df, review_df], axis=1)
    
    print("=== TEXT VECTORIZATION COMPLETE ===")
    print(f"Combined TF-IDF features: {tfidf_df.shape[1]}")
    print(f"Review TF-IDF features: {review_df.shape[1]}")
    print(f"Total text features: {text_features_combined.shape[1]}")
    print(f"NaNs in text features: {text_features_combined.isna().sum().sum()}")
    
    return text_features_combined, tfidf_combined, tfidf_reviews

# Create text features
text_features, tfidf_vectorizer, review_vectorizer = create_text_features(df_clean)

# ========================================
# STEP 6: CREATE FINAL FEATURE MATRICES (NO NaNs)
# ========================================

def create_final_feature_matrices(df_clean):
    """Create final feature matrices with no NaN values"""
    
    # 1. Numerical features (already scaled and imputed)
    numerical_features = df_clean[['rating_imputed_scaled', 'cost_imputed_scaled', 
                                  'log_votes_scaled', 'popularity_score_scaled']].copy()
    
    # 2. Content-based features
    content_features = pd.concat([
        numerical_features,
        service_features,
        location_encoded,
        rest_type_encoded,
        cuisine_df
    ], axis=1)
    
    # 3. Text-based features
    text_based_features = pd.concat([
        numerical_features[['rating_imputed_scaled', 'popularity_score_scaled']],  # Basic context
        text_features
    ], axis=1)
    
    # 4. Hybrid features (combination of all)
    hybrid_features = pd.concat([
        numerical_features,
        service_features,
        location_encoded.iloc[:, :10],  # Top 10 locations to control size
        rest_type_encoded.iloc[:, :5],   # Top 5 restaurant types
        cuisine_df.iloc[:, :15],         # Top 15 cuisines
        text_features.iloc[:, :20]       # Top 20 text features
    ], axis=1)
    
    # 5. Basic info for reference
    basic_info = df_clean[['name', 'address', 'location', 'cuisines', 
                          'rating_imputed', 'cost_imputed', 'rest_type', 
                          'online_order', 'book_table']].copy()
    
    # Rename columns for clarity
    basic_info = basic_info.rename(columns={
        'rating_imputed': 'rating',
        'cost_imputed': 'cost_for_two'
    })
    
    # Final verification - check for NaNs
    print("=== FINAL FEATURE MATRICES ===")
    print(f"Content features shape: {content_features.shape}")
    print(f"Text features shape: {text_based_features.shape}")
    print(f"Hybrid features shape: {hybrid_features.shape}")
    print(f"Basic info shape: {basic_info.shape}")
    
    print(f"\nNaN CHECK:")
    print(f"NaNs in content_features: {content_features.isna().sum().sum()}")
    print(f"NaNs in text_based_features: {text_based_features.isna().sum().sum()}")
    print(f"NaNs in hybrid_features: {hybrid_features.isna().sum().sum()}")
    print(f"NaNs in basic_info: {basic_info.isna().sum().sum()}")
    
    # Final cleanup - replace any remaining NaNs with 0
    content_features = content_features.fillna(0)
    text_based_features = text_based_features.fillna(0)
    hybrid_features = hybrid_features.fillna(0)
    
    print(f"\nAFTER FINAL CLEANUP:")
    print(f"NaNs in content_features: {content_features.isna().sum().sum()}")
    print(f"NaNs in text_based_features: {text_based_features.isna().sum().sum()}")
    print(f"NaNs in hybrid_features: {hybrid_features.isna().sum().sum()}")
    
    return content_features, text_based_features, hybrid_features, basic_info

# Create final feature matrices
content_features, text_features_final, hybrid_features, basic_info = create_final_feature_matrices(df_clean)

# ========================================
# STEP 7: SAVE PROCESSED DATA
# ========================================

print("=== SAVING PROCESSED DATA ===")

# Save all processed data
basic_info.to_csv('processed_restaurant_data.csv', index=False)
content_features.to_csv('content_features.csv', index=False)
text_features_final.to_csv('text_features.csv', index=False)
hybrid_features.to_csv('hybrid_features.csv', index=False)

# Save encoders and transformers
import pickle

with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(feature_scaler, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

with open('cuisine_encoder.pkl', 'wb') as f:
    pickle.dump(cuisine_encoder, f)

print("Feature engineering complete! Files saved:")
print("- processed_restaurant_data.csv")
print("- content_features.csv")
print("- text_features.csv") 
print("- hybrid_features.csv")
print("- feature_scaler.pkl")
print("- tfidf_vectorizer.pkl")
print("- cuisine_encoder.pkl")

# Display final summary
print(f"\n=== FINAL SUMMARY ===")
print(f"Total restaurants processed: {len(basic_info)}")
print(f"Content features: {content_features.shape[1]} dimensions")
print(f"Text features: {text_features_final.shape[1]} dimensions") 
print(f"Hybrid features: {hybrid_features.shape[1]} dimensions")
print(f"All features are NaN-free and ready for modeling!")