import streamlit as st
import pickle
import joblib
import gzip
import pandas as pd
from PIL import Image
import requests
from io import BytesIO

# Add caching for data and model loading
@st.cache_data
def load_data():
    raw_df = pd.read_csv('capetown_airbnb_df.csv.gz', compression='gzip')
    return raw_df

@st.cache_resource
def load_recommendation_model():
    return joblib.load('baseline_model.pkl')


# Initialize the app
st.title("Airbnb Recommender System")
st.info("Get personalized Airbnb recommendations based on your user profile")

# Load data and model
baseline_model = load_recommendation_model()
raw_df = load_data()

def recommend_airbnbs(user_id, listings_df, final_model):
    """
    Recommend 5 personalized Cape Town Airbnb listings using Streamlit
    """
    try:
        # Get listings already rated by the user
        user_listings = listings_df[listings_df['reviewer_id'] == user_id]['id_x'].unique()
        
        # Get all listings
        all_listings = listings_df['id_x'].unique()
        
        # Identify listings not yet rated by the user
        listings_to_predict = list(set(all_listings) - set(user_listings))
        
        # Create matrix for predictions
        user_listing_pairs = [(user_id, listing_id, 0) for listing_id in listings_to_predict]
        
        # Get predictions
        predictions = final_model.test(user_listing_pairs)
        
        # Get top 5 recommendations
        top_5_recs = sorted(predictions, key=lambda x: x.est, reverse=True)[:5]
        
        st.header("Your Personalized Recommendations")
        
        # Display recommendations in the main area
        for idx, rec in enumerate(top_5_recs, 1):
            # Get listing details
            listing = listings_df[listings_df['id_x'] == rec.iid].iloc[0]
            
            st.markdown(f"### Recommendation #{idx}: {listing['name']}")
            
            # Create three columns for the layout
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Main listing details
                st.markdown(f"**Description:** {listing['description']}")
                st.markdown(f"**Price:** R{listing['price']} per night")
                st.markdown(f"**Location:** {listing['neighbourhood_cleansed']}")
                st.markdown(f"**Property Type:** {listing['property_type']}")
                
                # Additional details section
                st.markdown("**Property Details:**")
                st.markdown(f"""
                * Accommodates: {listing['accommodates']} guests
                * Bedrooms: {listing['bedrooms']}
                * Bathrooms: {listing['bathrooms']}
                * Rating: {listing['avg_rating']}/5 ({listing['number_of_reviews']} reviews)
                """)
                
                # Add "View Listing" button
                if st.button(f"View Listing #{idx}", key=f"btn_{idx}"):
                    # Display image
                    response = requests.get(listing['picture_url'])
                    img = Image.open(BytesIO(response.content))
                    st.image(img, use_column_width=True)
                    st.markdown(f"[Open Airbnb Listing]({listing['listing_url']})")
            
            with col2:
                try:
                    # Display image
                    response = requests.get(listing['picture_url'])
                    img = Image.open(BytesIO(response.content))
                    st.image(img, use_column_width=True)
                except Exception as e:
                    st.error("Unable to load image")
            
            # Add a divider between recommendations
            st.divider()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def main():
    # Create a container for the input section
    with st.container():
        # Create columns for better layout
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.subheader("Enter Your Details")
            user_id = st.number_input(
                "User ID:",
                min_value=1,
                value=1412033,
                help="Enter your user ID to get personalized recommendations"
            )
            
            if st.button("Get Recommendations", type="primary"):
                recommend_airbnbs(user_id, raw_df, baseline_model)

if __name__ == "__main__":
    main()