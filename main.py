import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Data
df = pd.read_csv('gym_data.csv')

# 2. Features and Target
X = df[['Age', 'Duration', 'Avg_Heart_Rate', 'Gender']]
y = df['Calories_Burned']

# 3. Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predictions & Evaluation
y_pred = model.predict(X_test)
print(f"R-squared Score: {r2_score(y_test, y_pred):.2f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

# 6. Visualization
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Calories')
plt.ylabel('Predicted Calories')
plt.title('Actual vs Predicted Calories Burned')
plt.show()
