# House Price Prediction

## Introduction
This project aims to predict house prices using various features such as the number of bedrooms, bathrooms, living area square footage, lot size, number of floors, age of the house, and its condition. The goal is to demonstrate a typical machine learning workflow, from data generation and exploratory data analysis to model training, evaluation, and prediction.

## Features
The dataset for this project includes the following features:

* **`bedrooms`**: Number of bedrooms in the house.
* **`bathrooms`**: Number of bathrooms in the house.
* **`sqft_living`**: Square footage of the living area.
* **`sqft_lot`**: Square footage of the lot.
* **`floors`**: Number of floors in the house.
* **`age`**: Age of the house in years.
* **`condition`**: Condition of the house (a rating from 1 to 5, where 5 is excellent).
* **`price`**: The target variable, representing the house price.

## Technology Used
The project utilizes the following Python libraries for data manipulation, analysis, and machine learning:

* **`pandas`**: For data manipulation and analysis.
* **`numpy`**: For numerical operations.
* **`matplotlib`**: For creating static, animated, and interactive visualizations.
* **`seaborn`**: For statistical data visualization.
* **`scikit-learn`**: A comprehensive machine learning library, specifically used for:
    * `train_test_split`: For splitting data into training and testing sets.
    * `LinearRegression`: For implementing the Linear Regression model.
    * `RandomForestRegressor`: For implementing the Random Forest Regression model.
    * `mean_squared_error`, `r2_score`: For evaluating model performance.
    * `StandardScaler`: For feature scaling.

## Dataset
A synthetic dataset is generated for this project with 1000 samples. The `price` is calculated based on a linear combination of the other features with added random noise to simulate real-world variability.


## Usage

### 1. Import Libraries
All necessary libraries are imported at the beginning of the script.

### 2. Create Dataset
A synthetic dataset is created programmatically for demonstration purposes. This allows for reproducible results and easy experimentation without requiring external data files.

### 3. Exploratory Data Analysis (EDA)
Basic EDA is performed to understand the dataset's structure and characteristics:
* `df.info()`: Provides a summary of the DataFrame, including data types and non-null counts.
* `df.describe()`: Generates descriptive statistics of the numerical features.

### 4. Data Visualization
Various plots are generated to visualize the data and understand relationships between features and the target variable:
* **Price Distribution**: A histogram showing the distribution of house prices.
* **Correlation Heatmap**: A heatmap displaying the correlation matrix between all features, highlighting strong relationships.
* **Price vs. Square Feet Living**: A scatter plot showing the relationship between living area and price.
* **Price vs. Bedrooms**: A box plot illustrating the price distribution across different numbers of bedrooms.
* **Price vs. Age**: A scatter plot showing the relationship between house age and price.
* **Price vs. Condition**: A box plot illustrating the price distribution across different condition ratings.

### 5. Data Splitting and Scaling
The dataset is split into training and testing sets (80% training, 20% testing). Feature scaling (Standardization) is applied to the features using `StandardScaler` to normalize the data, which can improve the performance of some models.

### 6. Model Training
Two regression models are trained on the scaled training data:
* **Linear Regression Model**: A simple linear model for predicting house prices.
* **Random Forest Model**: An ensemble learning method that builds multiple decision trees and merges them to get a more accurate and stable prediction.

### 7. Prediction and Model Evaluation
Both trained models are used to make predictions on the test set. Model performance is evaluated using:
* **Root Mean Squared Error (RMSE)**: Measures the average magnitude of the errors.
* **R-squared ($R^2$) Score**: Represents the proportion of the variance in the dependent variable that is predictable from the independent variables.
* **Mean Absolute Error (MAE)**: Measures the average magnitude of the absolute errors.

**Evaluation Results:**

Linear Regression Performance:

    RMSE: $19,255.01

    R² Score: 0.9636

    Mean Absolute Error: $15,151.37

Random Forest Performance:

    RMSE: $23,323.73

    R² Score: 0.9466

    Mean Absolute Error: $18,245.20


### 8. Model Visualization
Scatter plots are generated to visualize the actual versus predicted prices for both models, along with a red dashed line representing perfect predictions.

### 9. Making Prediction for New Data
The trained models can be used to predict the price of a new, unseen house based on its features. An example prediction is shown for a hypothetical new house.

    **Example Prediction:**

    Linear Regression: $263,937.51

    Random Forest: $265,369.15



