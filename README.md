# Housing-linear-regression
House Price Prediction using Linear Regression
# Task 3 - Linear Regression
This project predicts house prices using area with the help of Linear Regression.

## Tools Used
- Python
- Pandas
- Scikit-learn
- Matplotlib

## Steps
1. Load data
2. Train linear regression model
3. Predict prices
4. Evaluate using MAE, MSE, R²
5. Visualize prediction

Dataset: [Housing Price Prediction on Kaggle](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)
This project focuses on implementing Linear Regression to predict house prices based on a key feature: the area of the house. The goal is to understand how linear regression models work and how they can be used to predict real-world numerical values by training on historical data.

The dataset used is publicly available on Kaggle and contains information about houses, including features such as area, bedrooms, bathrooms, stories, and price. For this task, we implement both Simple Linear Regression (using area only) and optionally Multiple Linear Regression (using more features) to explore how these models learn and make predictions.

🎯 Objective
Learn and implement Simple Linear Regression

Understand the relationship between house area and price

Train a machine learning model using scikit-learn

Evaluate model performance using common metrics: MAE, MSE, R² score

Visualize predictions using a regression line

🧰 Tools and Technologies
Python: Programming language

Pandas: For data handling and analysis

NumPy: For numerical computations

Scikit-learn: To build and evaluate the regression model

Matplotlib: To visualize the predictions and regression line

Jupyter Notebook or Google Colab: For writing and executing code

🪜 Steps Followed
Data Import: The dataset is read using pandas.read_csv() and basic information is displayed to understand the structure.

Feature Selection: The area column is selected as the input (independent variable), and price as the output (dependent variable).

Train-Test Split: The data is split into training and testing sets using train_test_split() to evaluate model performance on unseen data.

Model Training: A Linear Regression model is created using LinearRegression() from sklearn.linear_model and trained on the training data.

Prediction: The model is used to predict prices on the test data.

Evaluation:

MAE (Mean Absolute Error): Average absolute difference between actual and predicted prices.

MSE (Mean Squared Error): Squares the differences before averaging, penalizing larger errors more.

R² Score: Represents how well the model explains the variation in the data (1 means perfect prediction).

Visualization: A scatter plot is created for actual values and a line plot for predicted values, to visually assess the model’s performance.

📈 Results
The model was able to learn the trend between house area and price and provided predictions with acceptable accuracy, considering it was trained on a simple dataset. The evaluation metrics helped understand the model's performance numerically, while the plot illustrated the prediction visually.

🧠 Key Learnings
Understanding of how linear regression works for numeric predictions.

Hands-on experience in training and evaluating regression models.

Importance of splitting data and using metrics to assess model quality.

Visualization helps in interpreting model predictions clearly.


