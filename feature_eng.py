# %% [markdown]
# ## Import Library

# %%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import re
import ast

# Load the dataset
df = pd.read_csv(r'D:\development\recommendation_system\data\zomato.csv').sample(frac=0.3, random_state=42)
print(f"Original dataset shape: {df.shape}")

# %%
df.isnull().sum()

# %%
df.head()

# %% [markdown]
# ### Data Cleaning

# %%
df.rate.value_counts()

# %%
df.phone.head(10)

# %%
# Create a copy for feature engineering
df_clean = df.copy()

# 1. Clean rating column
def clean_rating(rate):
    if pd.isna(rate) or rate in ['NEW', '-', 'nan']:
        return np.nan
    try:
        return float(rate.split('/')[0])
    except:
        return np.nan

df_clean['rating'] = df_clean['rate'].apply(clean_rating)

# 2. Clean cost column
def clean_cost(cost):
    if pd.isna(cost):
        return np.nan
    try:
        # Remove commas and currency symbols
        cost_str = str(cost).replace(',', '').replace('₹', '').strip()
        return float(cost_str)
    except:
        return np.nan

df_clean['cost_for_two'] = df_clean['approx_cost(for two people)'].apply(clean_cost)

# 3. Clean phone numbers
# df_clean['phone_clean'] = df_clean['phone'].str.replace(r'\r\n.*', '', regex=True)
# Fill NaN with empty string, then split multiple numbers
df_clean['phone_clean'] = (
    df['phone']
    .fillna('Not Specified')                               # handle NaN
    .str.split(r'\r\n|\r\r\n')                # split on line breaks
)

# Keep only the first number (if you don’t want a list)
df_clean['phone_clean'] = df_clean['phone_clean'].str[0]
# 4. Handle missing values with meaningful defaults
df_clean.fillna({
    'location': 'Unknown',
    'rest_type': 'Not Specified',
    'cuisines': 'Not Specified',
    'dish_liked': ''
}, inplace=True)

print("=== AFTER CLEANING ===")
print(f"Dataset shape: {df_clean.shape}")
print(f"Missing ratings: {df_clean['rating'].isna().sum()}")
print(f"Missing costs: {df_clean['cost_for_two'].isna().sum()}")

# %%
df_clean.phone_clean.head(10),df_clean.phone_clean.isnull().sum()

# %% [markdown]
# ### Text Feature Engineering

# %%
# 1. Process cuisines (convert to list format)
def process_cuisines(cuisines):
    if pd.isna(cuisines) or cuisines == 'Not Specified':
        return []
    return [cuisine.strip().lower() for cuisine in cuisines.split(',')]

df_clean['cuisine_list'] = df_clean['cuisines'].apply(process_cuisines)

# 2. Process dishes liked
def process_dishes(dishes):
    if pd.isna(dishes) or dishes == '':
        return []
    return [dish.strip().lower() for dish in dishes.split(',')]

df_clean['dish_list'] = df_clean['dish_liked'].apply(process_dishes)

# 3. Create combined text features for similarity
df_clean['combined_features'] = (
    df_clean['cuisines'].fillna('') + ' ' + 
    df_clean['dish_liked'].fillna('') + ' ' +
    df_clean['rest_type'].fillna('')
)

# 4. Process reviews (extract text from reviews_list)
def extract_review_text(reviews_list):
    if pd.isna(reviews_list) or reviews_list == '[]':
        return ''
    try:
        reviews = ast.literal_eval(reviews_list)
        review_texts = []
        for review in reviews:
            if isinstance(review, tuple) and len(review) > 1:
                review_texts.append(review[1])
        return ' '.join(review_texts)
    except:
        return ''

df_clean['review_text'] = df_clean['reviews_list'].apply(extract_review_text)

print("=== TEXT PROCESSING COMPLETE ===")
print(f"Average cuisines per restaurant: {df_clean['cuisine_list'].apply(len).mean():.1f}")
print(f"Restaurants with dishes mentioned: {(df_clean['dish_list'].apply(len) > 0).sum()}")

# %%
df_clean.isnull().sum()

# %% [markdown]
# ### Categorical Feature Engineering

# %%
# 1. Binary encode yes/no features
df_clean['online_order_binary'] = (df_clean['online_order'] == 'Yes').astype(int)
df_clean['book_table_binary'] = (df_clean['book_table'] == 'Yes').astype(int)

# 2. Create location clusters (group less frequent locations)
location_counts = df_clean['location'].value_counts()
popular_locations = location_counts[location_counts >= 50].index
df_clean['location_grouped'] = df_clean['location'].apply(
    lambda x: x if x in popular_locations else 'Other'
)

# 3. Restaurant type grouping
rest_type_counts = df_clean['rest_type'].value_counts()
popular_rest_types = rest_type_counts[rest_type_counts >= 30].index
df_clean['rest_type_grouped'] = df_clean['rest_type'].apply(
    lambda x: x if x in popular_rest_types else 'Other'
)

# 4. Create price categories
def categorize_price(cost):
    if pd.isna(cost):
        return 'Unknown'
    elif cost <= 300:
        return 'Budget'
    elif cost <= 600:
        return 'Mid-range'
    elif cost <= 1200:
        return 'Expensive'
    else:
        return 'Premium'

df_clean['price_category'] = df_clean['cost_for_two'].apply(categorize_price)

# 5. Create rating categories
def categorize_rating(rating):
    if pd.isna(rating):
        return 'Unrated'
    elif rating < 3.0:
        return 'Poor'
    elif rating < 3.5:
        return 'Average'
    elif rating < 4.0:
        return 'Good'
    else:
        return 'Excellent'

df_clean['rating_category'] = df_clean['rating'].apply(categorize_rating)

print("=== CATEGORICAL ENCODING COMPLETE ===")
print(f"Location groups: {df_clean['location_grouped'].nunique()}")
print(f"Restaurant type groups: {df_clean['rest_type_grouped'].nunique()}")
print(f"Price categories: {df_clean['price_category'].value_counts()}")

# %%
df_clean.isnull().sum()

# %% [markdown]
# ###  Numerical Feature Engineering

# %%
# # 1. Handle missing numerical values
# df_clean['rating'].fillna(df_clean['rating'].median(), inplace=True)
# df_clean['cost_for_two'].fillna(df_clean['cost_for_two'].median(), inplace=True)
# df_clean['votes'].fillna(0, inplace=True)

# # 2. Create derived numerical features
# df_clean['rating_votes_ratio'] = df_clean['rating'] * np.log1p(df_clean['votes'])
# df_clean['cost_per_rating'] = df_clean['cost_for_two'] / (df_clean['rating'] + 0.1)  # Avoid division by zero
# df_clean['popularity_score'] = np.log1p(df_clean['votes']) * df_clean['rating']

# # 3. Normalize numerical features
# scaler = StandardScaler()
# numerical_features = ['rating', 'cost_for_two', 'votes', 'rating_votes_ratio', 'popularity_score']
# df_clean[['rating_scaled', 'cost_scaled', 'votes_scaled', 'ratio_scaled', 'popularity_scaled']] = \
#     scaler.fit_transform(df_clean[numerical_features])

# print("=== NUMERICAL FEATURE ENGINEERING COMPLETE ===")
# print("New numerical features created:")
# for feature in ['rating_votes_ratio', 'cost_per_rating', 'popularity_score']:
#     print(f"  {feature}: mean={df_clean[feature].mean():.3f}, std={df_clean[feature].std():.3f}")

# %%
from sklearn.preprocessing import StandardScaler
import numpy as np

# 1. Handle missing numerical values (no inplace warnings)
df_clean['rating'] = df_clean['rating'].fillna(df_clean['rating'].median())
df_clean['cost_for_two'] = df_clean['cost_for_two'].fillna(df_clean['cost_for_two'].median())
df_clean['votes'] = df_clean['votes'].fillna(0)

# 2. Create derived numerical features
df_clean['rating_votes_ratio'] = df_clean['rating'] * np.log1p(df_clean['votes'])
df_clean['cost_per_rating'] = df_clean['cost_for_two'] / (df_clean['rating'] + 0.1)  # Avoid division by zero
df_clean['popularity_score'] = np.log1p(df_clean['votes']) * df_clean['rating']

# 3. Normalize numerical features
scaler = StandardScaler()
numerical_features = ['rating', 'cost_for_two', 'votes', 'rating_votes_ratio', 'popularity_score']
df_clean[['rating_scaled', 'cost_scaled', 'votes_scaled', 'ratio_scaled', 'popularity_scaled']] = (
    scaler.fit_transform(df_clean[numerical_features])
)

# 4. Summary
print("=== NUMERICAL FEATURE ENGINEERING COMPLETE ===")
print("New numerical features created:")
for feature in ['rating_votes_ratio', 'cost_per_rating', 'popularity_score']:
    print(f"  {feature}: mean={df_clean[feature].mean():.3f}, std={df_clean[feature].std():.3f}")


# %%
df_clean.isnull().sum()

# %% [markdown]
# ### Advanced Feature Engineering

# %%
# 1. Cuisine similarity features using MultiLabelBinarizer
mlb_cuisine = MultiLabelBinarizer()
cuisine_encoded = mlb_cuisine.fit_transform(df_clean['cuisine_list'])
cuisine_df = pd.DataFrame(cuisine_encoded, columns=mlb_cuisine.classes_)

# Add prefix to avoid column name conflicts
cuisine_df.columns = ['cuisine_' + col for col in cuisine_df.columns]

# 2. TF-IDF features for text similarity
tfidf_combined = TfidfVectorizer(
    max_features=100,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2
)
tfidf_features = tfidf_combined.fit_transform(df_clean['combined_features'])
tfidf_df = pd.DataFrame(
    tfidf_features.toarray(), 
    columns=['tfidf_' + str(i) for i in range(tfidf_features.shape[1])]
)

# 3. Location encoding (one-hot for grouped locations)
location_encoded = pd.get_dummies(df_clean['location_grouped'], prefix='location')

# 4. Restaurant type encoding
rest_type_encoded = pd.get_dummies(df_clean['rest_type_grouped'], prefix='rest_type')

# 5. Service features
service_df = df_clean[['online_order_binary', 'book_table_binary']].copy()

print("=== ADVANCED FEATURE ENGINEERING COMPLETE ===")
print(f"Cuisine features: {cuisine_df.shape[1]}")
print(f"TF-IDF features: {tfidf_df.shape[1]}")
print(f"Location features: {location_encoded.shape[1]}")
print(f"Restaurant type features: {rest_type_encoded.shape[1]}")

# %% [markdown]
# ### Create Final Feature Matrix

# %%
# Combine all features for similarity calculation
feature_matrices = {
    'basic_info': df_clean[['name', 'address', 'phone_clean', 'location', 'cuisines']],
    'numerical': df_clean[['rating_scaled', 'cost_scaled', 'votes_scaled', 'popularity_scaled']],
    'categorical': df_clean[['price_category', 'rating_category']],
    'services': service_df,
    'cuisines': cuisine_df,
    'location': location_encoded,
    'rest_type': rest_type_encoded,
    'tfidf': tfidf_df
}

# Create the main feature matrix for similarity (excluding basic info)
similarity_features = pd.concat([
    feature_matrices['numerical'],
    feature_matrices['services'],
    feature_matrices['cuisines'],
    feature_matrices['location'],
    feature_matrices['rest_type'],
    feature_matrices['tfidf']
], axis=1)

# Save processed data
df_clean.to_csv('processed_restaurant_data.csv', index=False)
similarity_features.to_csv('similarity_features.csv', index=False)

print("=== FEATURE MATRIX CREATION COMPLETE ===")
print(f"Final feature matrix shape: {similarity_features.shape}")
print(f"Features per category:")
for category, matrix in feature_matrices.items():
    if category != 'basic_info':
        print(f"  {category}: {matrix.shape[1]} features")

# Display feature importance/variance
print("\n=== FEATURE STATISTICS ===")
feature_variance = similarity_features.var().sort_values(ascending=False)
print("Top 10 features by variance:")
print(feature_variance.head(10))

# %% [markdown]
# ### Create Different Feature Sets for Experimentation

# %%
# Create multiple feature sets for different recommendation approaches

# 1. Content-based features (cuisine + location + type + price)
content_features = pd.concat([
    feature_matrices['numerical'][['rating_scaled', 'cost_scaled']],
    feature_matrices['services'],
    feature_matrices['cuisines'],
    feature_matrices['location'],
    feature_matrices['rest_type']
], axis=1)

# 2. Text-based features (TF-IDF + rating + popularity)
text_features = pd.concat([
    feature_matrices['numerical'][['rating_scaled', 'popularity_scaled']],
    feature_matrices['tfidf']
], axis=1)

# 3. Hybrid features (all combined with weights)
hybrid_features = similarity_features.copy()

# Apply feature weights (can be tuned based on domain knowledge)
weight_config = {
    'cuisine_weight': 2.0,
    'location_weight': 1.5,
    'price_weight': 1.2,
    'rating_weight': 1.8,
    'text_weight': 1.0
}

# Apply weights to cuisine features
cuisine_cols = [col for col in hybrid_features.columns if col.startswith('cuisine_')]
hybrid_features[cuisine_cols] *= weight_config['cuisine_weight']

# Apply weights to location features
location_cols = [col for col in hybrid_features.columns if col.startswith('location_')]
hybrid_features[location_cols] *= weight_config['location_weight']

# Apply weights to rating
hybrid_features['rating_scaled'] *= weight_config['rating_weight']

# Save different feature sets
content_features.to_csv('content_features.csv', index=False)
text_features.to_csv('text_features.csv', index=False)
hybrid_features.to_csv('hybrid_features.csv', index=False)

print("=== MULTIPLE FEATURE SETS CREATED ===")
print(f"Content-based features: {content_features.shape}")
print(f"Text-based features: {text_features.shape}")
print(f"Hybrid features: {hybrid_features.shape}")

# %%



