import streamlit as st 
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add caching for data loading
@st.cache_data
def load_data():
    capetown_aggregated_df = pd.read_csv("capetown_aggregated_df.csv")
    raw_df = pd.read_csv('capetown_airbnb_df.csv.gz', compression='gzip')
    raw_df['date'] = pd.to_datetime(raw_df['date'])
    raw_df['month'] = raw_df['date'].dt.month
    return capetown_aggregated_df, raw_df

@st.cache_resource
def load_model():
    return joblib.load('polyregression_model.pkl')

# Initialize the app
st.set_page_config(page_title="Airbnb Recommender System", layout="wide")
st.title("Airbnb Recommender System App")
st.info("This app builds an Airbnb recommender System")

# Load data and model using cached functions
loaded_model = load_model()
capetown_aggregated_df, raw_df = load_data()

# Create tabs for better organization and performance
tab1, tab2 = st.tabs(["Data & Predictions", "Visualizations"])

with tab1:
    # Data expander
    with st.expander('Data'):
        st.write('Raw data')
        st.dataframe(capetown_aggregated_df)  # Using st.dataframe instead of st.write for better performance

with tab2:
    # Visualizations expander
    col1, col2 = st.columns(2)
    
    with col1:
        st.write('**Price v. Average Rating**')
        st.scatter_chart(
            data=capetown_aggregated_df,
            x='price',
            y='avg_rating',
            color='#ffaa0088'
        )
        
        st.write('**Property type Distribution**')
        # Optimize property type calculations
        @st.cache_data
        def get_property_type_data():
            property_counts = raw_df['property_type'].value_counts()
            top_property_types = property_counts.nlargest(10).index
            avg_price_per_property_type = raw_df[raw_df['property_type'].isin(top_property_types)] \
                .groupby('property_type')['price'].mean().reset_index()
            return avg_price_per_property_type.sort_values(by='price', ascending=False)
            
        st.bar_chart(
            data=get_property_type_data(),
            x='property_type',
            y='price'
        )
    
    with col2:
        st.write('**Price Distribution**')
        fig, ax = plt.subplots(figsize=(8,6))
        ax.hist(data=capetown_aggregated_df, x='price', bins=20, color='lightblue', edgecolor='black')
        st.pyplot(fig)
        
        st.write('**Monthly Price Distribution**')
        @st.cache_data
        def get_monthly_price_data():
            return raw_df.groupby('month')['price'].mean().reset_index()
            
        st.line_chart(
            data=get_monthly_price_data(),
            x='month',
            y='price'
        )

    st.write('**Area Visualization**')
    st.map(
        data=raw_df,
        latitude='latitude',
        longitude='longitude',
        use_container_width=True
    )

# Optimize the suggest_price function
@st.cache_data
def calculate_frequency_encodings(neighbourhood, property_type, df):
    neighbourhood_freq = df[
        df['neighbourhood_cleansed'] == neighbourhood
    ]['neighbourhood_cleansed_freq'].mean()
    
    property_freq = df[
        df['property_type'] == property_type
    ]['property_type_freq'].mean()
    
    return neighbourhood_freq, property_freq

def suggest_price(model, capetown_aggregated_df):
    with st.sidebar:
        st.header('Property Details')
        
        # Use columns for compact layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Host Information
            st.subheader("Host Info")
            host_id = st.number_input("Host ID:", min_value=0)
            host_response_rate = st.number_input("Response rate:", 0.0, 1.0, 0.9)
            host_is_superhost = st.radio("Superhost?", ["yes", "no"])
            host_listings_count = st.number_input("Listings count:", 1)

        with col2:
            # Property Details
            st.subheader("Property Info")
            property_type = st.selectbox("Type:", capetown_aggregated_df['property_type'].unique())
            accommodates = st.number_input("Accommodates:", 1, 20, 2)
            bathrooms = st.number_input("Bathrooms:", 0.0, 10.0, 1.0)
            bedrooms = st.number_input("Bedrooms:", 0.0, 10.0, 1.0)
            beds = st.number_input("Beds:", 1, 20, 1)

        # Location and Ratings
        neighbourhood_cleansed = st.selectbox("Neighbourhood:", capetown_aggregated_df['neighbourhood_cleansed'].unique())
        avg_rating = st.number_input("Rating:", 0.0, 5.0, 4.5)
        number_of_reviews = st.number_input("Reviews:", 0, 1000, 10)

        if st.button("Calculate Price"):
            # Validate and calculate
            if neighbourhood_cleansed in capetown_aggregated_df['neighbourhood_cleansed'].values and \
               property_type in capetown_aggregated_df['property_type'].values:
                
                # Get frequency encodings
                neighbourhood_freq, property_freq = calculate_frequency_encodings(
                    neighbourhood_cleansed, 
                    property_type, 
                    capetown_aggregated_df
                )
                
                # Create prediction DataFrame
                property_details = pd.DataFrame({
                    'host_id': [host_id],
                    'host_response_rate': [host_response_rate],
                    'host_is_superhost': [1 if host_is_superhost.lower() == 'yes' else 0],
                    'host_listings_count': [host_listings_count],
                    'accommodates': [accommodates],
                    'bathrooms': [bathrooms],
                    'bedrooms': [bedrooms],
                    'beds': [beds],
                    'avg_rating': [avg_rating],
                    'number_of_reviews': [number_of_reviews],
                    'neighbourhood_cleansed_freq': [neighbourhood_freq],
                    'property_type_freq': [property_freq]
                })
                
                # Make prediction
                log_price = model.predict(property_details)
                price = np.exp(log_price[0])
                return round(price, 2)
            else:
                st.error("Invalid neighbourhood or property type")
                return None

suggested_price = suggest_price(loaded_model, capetown_aggregated_df)
st.write(f"Suggested price: ZAR {suggested_price}")