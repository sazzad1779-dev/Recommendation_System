# app.py
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# from recommenders import ContentBasedRecommender, TextBasedRecommender,HybridRecommender
from src.text_based_recommender import TextBasedRecommender
from src.content_base_recommender import ContentBasedRecommender
from src.hybrid_recommender import HybridRecommender
# Load models and data
@st.cache_data
def load_data():
    try:
        df_clean = pd.read_csv('processed_data/processed_restaurant_data.csv')
        return df_clean
    except:
        st.error("Please ensure processed_restaurant_data.csv is available")
        return None

@st.cache_resource
def load_models():
    try:
        with open('trained_model/content_recommender.pkl', 'rb') as f:
            content_recommender = pickle.load(f)
        with open('trained_model/text_recommender.pkl', 'rb') as f:
            text_recommender = pickle.load(f)
        with open('trained_model/hybrid_recommender.pkl', 'rb') as f:
            hybrid_recommender = pickle.load(f)
        return content_recommender, text_recommender, hybrid_recommender
    except:
        st.error("Please ensure model files are available")
        return None, None, None

# Main app
def main():
    st.set_page_config(
        page_title="Restaurant Recommendation System",
        page_icon="🍽️",
        layout="wide"
    )
    
    st.title("🍽️ Restaurant Recommendation System")
    st.markdown("*Discover your next favorite restaurant in Bengaluru*")
    
    # Load data and models
    df_clean = load_data()
    content_recommender, text_recommender, hybrid_recommender = load_models()
    
    if df_clean is None or content_recommender is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["🏠 Home", "🔍 Get Recommendations", "📊 Analytics", "ℹ️ About"])
    
    if page == "🏠 Home":
        show_home_page(df_clean)
    elif page == "🔍 Get Recommendations":
        show_recommendations_page(df_clean, content_recommender, text_recommender, hybrid_recommender)
    elif page == "📊 Analytics":
        show_analytics_page(df_clean)
    elif page == "ℹ️ About":
        show_about_page()

def show_home_page(df_clean):
    st.header("Welcome to the Restaurant Recommendation System")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Restaurants", len(df_clean))
    
    with col2:
        st.metric("Unique Locations", df_clean['location'].nunique())
    
    with col3:
        st.metric("Average Rating", f"{df_clean['rating'].mean():.1f}")
    
    with col4:
        st.metric("Cuisines Available", df_clean['cuisines'].nunique())
    
    # Quick overview charts
    st.subheader("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top locations
        top_locations = df_clean['location'].value_counts().head(10)
        fig = px.bar(x=top_locations.values, y=top_locations.index, 
                     orientation='h', title="Top 10 Locations")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Rating distribution
        fig = px.histogram(df_clean, x='rating', nbins=20, 
                          title="Rating Distribution")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_recommendations_page(df_clean, content_recommender, text_recommender, hybrid_recommender):
    st.header("Get Restaurant Recommendations")
    
    # Restaurant selection
    restaurant_names = sorted(df_clean['name'].unique())
    selected_restaurant = st.selectbox("Select a restaurant:", restaurant_names)
    
    # Recommendation method
    method = st.radio("Choose recommendation method:", 
                     ["Hybrid (Recommended)", "Content-based", "Text-based"])
    
    # Number of recommendations
    n_recommendations = st.slider("Number of recommendations:", 1, 20, 10)
    
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            try:
                # Get recommendations based on method
                if method == "Hybrid (Recommended)":
                    recommendations = hybrid_recommender.get_hybrid_recommendations(
                        selected_restaurant, n_recommendations)
                    score_col = 'hybrid_score'
                elif method == "Content-based":
                    recommendations = content_recommender.get_recommendations(
                        selected_restaurant, n_recommendations)
                    score_col = 'similarity_score'
                else:  # Text-based
                    recommendations = text_recommender.get_recommendations(
                        selected_restaurant, n_recommendations)
                    score_col = 'similarity_score'
                
                if isinstance(recommendations, str):
                    st.error(recommendations)
                else:
                    # Display original restaurant info
                    original = df_clean[df_clean['name'] == selected_restaurant].iloc[0]
                    
                    st.subheader("Selected Restaurant")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Cuisine:** {original['cuisines']}")
                        st.write(f"**Location:** {original['location']}")
                    with col2:
                        st.write(f"**Rating:** {original['rating']}")
                        st.write(f"**Cost for Two:** ₹{original['cost_for_two']}")
                    with col3:
                        st.write(f"**Type:** {original['rest_type']}")
                        st.write(f"**Online Order:** {original['online_order']}")
                    
                    st.subheader("Recommended Restaurants")
                    
                    # Display recommendations
                    for idx, row in recommendations.iterrows():
                        with st.expander(f"{idx+1}. {row['name']} (Score: {row[score_col]:.3f})"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Cuisine:** {row['cuisines']}")
                                st.write(f"**Location:** {row['location']}")
                                st.write(f"**Rating:** {row['rating']}")
                            with col2:
                                st.write(f"**Cost for Two:** ₹{row['cost_for_two']}")
                                st.write(f"**Type:** {row['rest_type']}")
                                st.write(f"**Online Order:** {row['online_order']}")
                    
                    # Visualization
                    fig = px.bar(recommendations, x=score_col, y='name', 
                               orientation='h', title="Recommendation Scores")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")

def show_analytics_page(df_clean):
    st.header("Restaurant Analytics")
    
    # Filter options
    st.sidebar.subheader("Filters")
    selected_locations = st.sidebar.multiselect("Locations", 
                                               df_clean['location'].unique())
    
    # Filter data
    filtered_df = df_clean.copy()
    if selected_locations:
        filtered_df = filtered_df[filtered_df['location'].isin(selected_locations)]
    
    # Analytics visualizations
    tab1, tab2, tab3 = st.tabs(["Location Analysis", "Cuisine Analysis", "Price Analysis"])
    
    with tab1:
        if not filtered_df.empty:
            # Location-wise statistics
            location_stats = filtered_df.groupby('location').agg({
                'rating': 'mean',
                'cost_for_two': 'mean',
                'name': 'count'
            }).round(2)
            location_stats.columns = ['Avg Rating', 'Avg Cost', 'Restaurant Count']
            
            st.subheader("Location Statistics")
            st.dataframe(location_stats)
            
            # Visualization
            fig = make_subplots(rows=1, cols=2, 
                              subplot_titles=('Average Rating by Location', 
                                            'Average Cost by Location'))
            
            fig.add_trace(go.Bar(x=location_stats.index, 
                               y=location_stats['Avg Rating'],
                               name='Rating'), row=1, col=1)
            
            fig.add_trace(go.Bar(x=location_stats.index, 
                               y=location_stats['Avg Cost'],
                               name='Cost'), row=1, col=2)
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Cuisine analysis (simplified)
        st.subheader("Popular Cuisines")
        if not filtered_df.empty:
            # Extract all cuisines
            all_cuisines = []
            for cuisines in filtered_df['cuisines'].dropna():
                cuisine_list = [c.strip() for c in str(cuisines).split(',')]
                all_cuisines.extend(cuisine_list)
            
            cuisine_counts = pd.Series(all_cuisines).value_counts().head(15)
            
            fig = px.bar(x=cuisine_counts.values, y=cuisine_counts.index,
                        orientation='h', title="Top 15 Cuisines")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Price analysis
        st.subheader("Price Distribution")
        if not filtered_df.empty:
            fig = px.histogram(filtered_df, x='cost_for_two', nbins=30,
                             title="Cost Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Price vs Rating
            fig = px.scatter(filtered_df, x='cost_for_two', y='rating',
                           title="Price vs Rating Relationship")
            st.plotly_chart(fig, use_container_width=True)

def show_about_page():
    st.header("About This System")
    
    st.markdown("""
    ### Restaurant Recommendation System
    
    This system uses machine learning to recommend restaurants based on various features:
    
    #### Features Used:
    - **Cuisines**: Types of food served
    - **Location**: Restaurant location in Bengaluru
    - **Price Range**: Cost for two people
    - **Ratings**: User ratings and votes
    - **Restaurant Type**: Dining style (Casual, Fine Dining, etc.)
    - **Services**: Online ordering and table booking availability
    
    #### Recommendation Methods:
    1. **Content-based**: Based on restaurant features and characteristics
    2. **Text-based**: Based on reviews and text descriptions
    3. **Hybrid**: Combines both approaches for better recommendations
    
    #### Technology Stack:
    - **Python**: Core programming language
    - **Scikit-learn**: Machine learning algorithms
    - **Pandas**: Data manipulation and analysis
    - **Streamlit**: Web application framework
    - **Plotly**: Interactive visualizations
    
    ### How to Use:
    1. Go to the "Get Recommendations" page
    2. Select a restaurant you like
    3. Choose a recommendation method
    4. Get personalized restaurant suggestions!
    """)

if __name__ == "__main__":
    main()