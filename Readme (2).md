\# Salary Prediction using Linear Regression

This project demonstrates how to predict employee salaries based on
years of experience using the Linear Regression algorithm in Python. The
entire process is implemented in Google Colab for ease of use and
accessibility.

\## Table of Contents

\- \[Project Overview\](#project-overview) - \[Dataset\](#dataset) -
\[Dependencies\](#dependencies) - \[Installation\](#installation) -
\[Usage\](#usage) - \[Project Structure\](#project-structure) -
\[Results\](#results) - \[Contributing\](#contributing) -
\[License\](#license)

\## Project Overview

In this project, we aim to predict employee salaries based on their
years of experience. The problem is framed as a regression task, where
the target variable is the salary, and the input feature is the years of
experience. Linear Regression, a simple and effective machine learning
algorithm, is used for this purpose.

\## Dataset

The dataset used for this project contains two columns:

1\. \*\*YearsExperience\*\*: The number of years of professional
experience. 2. \*\*Salary\*\*: The annual salary corresponding to the
experience.

A sample dataset (\`salary_data.csv\`) is provided.

\*\*Sample Data\*\*: \| YearsExperience \| Salary \|
\|-----------------\|---------\| \| 1.1 \| 39343 \| \| 1.3 \| 46205 \|
\| 1.5 \| 37731 \| \| ... \| ... \|

\## Dependencies

The following Python libraries are required to run the project:

\- Python 3.x - Numpy - Pandas - Matplotlib - Scikit-learn - Google
Colab (for running the notebook online)

\## Installation

To set up this project in Google Colab, follow these steps:

1\. Open \[Google Colab\](https://colab.research.google.com/). 2. Upload
the notebook (\`Salary_Prediction.ipynb\`) and the dataset
(\`salary_data.csv\`). 3. Ensure the necessary dependencies are
installed in your Colab environment: \`\`\`python !pip install numpy
pandas matplotlib scikit-learn \`\`\`

\## Usage

1\. \*\*Data Preparation\*\*:  - Load the dataset using \`pandas\`.  -
Split the data into features (YearsExperience) and target (Salary).

2\. \*\*Train-Test Split\*\*:  - Divide the dataset into training and
testing sets using \`train_test_split\` from \`sklearn\`.

3\. \*\*Model Training\*\*:  - Train the Linear Regression model using
\`LinearRegression\` from \`sklearn\`.

4\. \*\*Model Evaluation\*\*:  - Predict the salaries on the test data.
 - Calculate metrics like Mean Squared Error (MSE) and R-squared (R²) to
evaluate the model.

5\. \*\*Visualization\*\*:  - Plot the regression line using
\`matplotlib\`.

\### Sample Code (inside Colab notebook): \`\`\`python \# Import
necessary libraries import numpy as np import pandas as pd import
matplotlib.pyplot as plt from sklearn.model_selection import
train_test_split from sklearn.linear_model import LinearRegression from
sklearn.metrics import mean_squared_error, r2_score

\# Load the dataset data = pd.read_csv('salary_data.csv')

\# Split the data into features and target variable X =
data\['YearsExperience'\].values.reshape(-1, 1) y = data\['Salary'\]

\# Split into training and testing data X_train, X_test, y_train, y_test
= train_test_split(X, y, test_size=0.2, random_state=42)

\# Initialize and train the model model = LinearRegression()
model.fit(X_train, y_train)

\# Predicting the test results y_pred = model.predict(X_test)

\# Visualizing the results plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, model.predict(X_train), color='red') plt.title('Years
of Experience vs Salary (Training Set)') plt.xlabel('Years of
Experience') plt.ylabel('Salary') plt.show()

\# Evaluation mse = mean_squared_error(y_test, y_pred) r2 =
r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}") print(f"R-Squared: {r2}") \`\`\`

\## Project Structure

\`\`\` ├── Salary_Prediction.ipynb \# Main notebook file ├──
salary_data.csv \# Dataset └── README.md \# This file \`\`\`

\## Results

The project provides insights into how years of experience affect
salary. The Linear Regression model fits a straight line to the data,
allowing us to make salary predictions for new employees based on their
experience.

Key metrics: - \*\*Mean Squared Error (MSE)\*\*: Indicates the average
squared difference between the predicted and actual values. - \*\*R²
Score\*\*: Measures how well the variance in the data is explained by
the model.

\## Contributing

Feel free to fork this repository, make changes, and submit a pull
request. Contributions are welcome!

\## License

This project is licensed under the MIT License.

---

This file provides a comprehensive overview of the project, guiding
users on how to set it up and use it effectively in Google Colab.
