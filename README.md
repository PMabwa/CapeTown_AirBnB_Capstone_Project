# Airbnb-Recommender-System

## Project Overview

This project builds a recommender system for Cape Town Airbnb listings to help hosts optimize pricing, occupancy, and guest satisfaction. Using data from [Inside Airbnb](https://insideairbnb.com/get-the-data/), the model suggests optimal prices based on listing features, guest sentiment, and occupancy patterns, helping hosts set competitive and profitable prices.

## Data Description

The dataset comprises information about Airbnb listings in Cape Town, focusing on features that influence pricing, occupancy, and guest satisfaction. Key data files include:

- **listings.csv**: Contains details about each listing, such as property type, location, amenities, and host information.
- **calendar.csv**: Shows the availability and pricing for each listing over time.
- **reviews.csv**: Provides guest feedback, which is used to derive sentiment scores.

### Key Columns

- **Property Details**: Includes `property_type`, `accommodates`, `bathrooms`, `bedrooms`, and `beds`.
- **Host Information**: Fields like `host_id`, `host_response_rate`, `host_is_superhost`, and `host_listings_count`.
- **Pricing and Occupancy**: Columns such as `price`, `availability`, and `number_of_reviews`.
- **Guest Sentiment**: Derived from guest comments using sentiment analysis to score the emotional tone of each review.
  
This structure enables the model to incorporate a wide range of factors influencing Airbnb's performance in Cape Town.

## Data Preprocessing

Data preprocessing involved several steps to prepare the dataset for modeling:

1. **Data Cleaning**:
   - Removed duplicate entries and irrelevant columns.
   - Addressed missing values through imputation for numerical columns and frequency encoding for categorical ones.

2. **Feature Engineering**:
   - Created new features such as `sentiment_score`, extracted from guest reviews using sentiment analysis.
   - Encoded categorical variables, applying frequency encoding to columns like `neighbourhood_cleansed` and `property_type` for improved model performance.

3. **Transformations**:
   - Log-transformed the `price` column to reduce skewness and approximate a normal distribution.

4. **Data Splitting**:
   - Split the data into training and testing sets, ensuring each listing appeared only once in the analysis.

These preprocessing steps allowed for better handling of categorical data and helped optimize model performance.

## Exploratory Data Analysis (EDA)

The EDA focused on understanding key trends and distributions in the dataset. Below are some visual insights generated:

1. **Price Distribution**: Showcasing the range and skewness of listing prices.  
   ![Price Distribution]([path/to/Price-distribution.png](https://github.com/PMabwa/airbnb-recommender-system/blob/Festus/images/Price-distribution.png))

2. **Property Type Breakdown**: An overview of listing types, such as entire homes, private rooms, etc.  
   ![Property Type Breakdown]([path/to/Property-type.png"](https://github.com/PMabwa/airbnb-recommender-system/blob/Festus/images/Property-type.png))

3. **Neighborhood Popularity**: Visualizing the distribution of listings across various neighborhoods.  
   ![Neighborhood Popularity]([path/to/Neighborhoods.png"](https://github.com/PMabwa/airbnb-recommender-system/blob/Festus/images/Neighborhoods.png))

4. **Occupancy Rates by Property Type**: Analyzing how occupancy varies among different property types.  
   ![Occupancy Rates by Property Type]([path/to/rating-property.png"](https://github.com/PMabwa/airbnb-recommender-system/blob/Festus/images/rating-property.png))

5. **Sentiment Score Distribution**: Analysis of guest review sentiments, highlighting the frequency of positive and negative feedback.  
   ![Sentiment Score Distribution]([path/to/sentiment-score.png](https://1drv.ms/i/c/5dbeccba30e4cdcb/EYxfGHT4ghVJm5YIw5CZZ4oB4zlGxb620YFLETrzrVyaNA?e=IU57t4))

These visualizations provided a foundation for understanding factors like pricing, guest sentiment, and listing characteristics that inform the recommendations and predictions.

## Modelling

| **Model**                          	| **Train RMSE** 	| **Test RMSE** 	| **R² Score** 	|
|------------------------------------	|----------------	|---------------	|--------------	|
| Baseline Model: Linear Regression  	| 0.67           	| 0.76          	| 0.5961       	|
| Linear Regression (with PCA)       	| 0.71           	| 0.79          	| 0.5721       	|
| Decision Tree Model                	| 0.61           	| 0.70          	| 0.6412       	|
| Random Forest                      	| 0.17           	| 0.57          	| 0.7408       	|
| KNN Regression Model               	| 0.56           	| 0.68          	| 0.6616       	|
| Tuned KNN Model (with Grid Search) 	| 0.54           	| 0.68          	| 0.6621       	|
| XGBoost Model                      	| 0.43           	| 0.55          	| 0.7579       	|
| LightGBM Model                     	| 0.32           	| 0.54          	| 0.7699       	|
| Neural Network Model               	| 0.56           	| 0.64          	| 0.6900       	|
