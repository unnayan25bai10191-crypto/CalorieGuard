import pandas as pd
import numpy as np

# Create synthetic data
data = {
    'Age': np.random.randint(18, 60, 200),
    'Duration': np.random.randint(20, 90, 200),
    'Avg_Heart_Rate': np.random.randint(100, 170, 200),
    'Gender': np.random.randint(0, 2, 200)
}
df = pd.DataFrame(data)

# Simple formula for calories burned
df['Calories_Burned'] = (df['Duration'] * 7.5) + (df['Avg_Heart_Rate'] * 0.4) + np.random.normal(0, 10, 200)

df.to_csv('gym_data.csv', index=False)
print("gym_data.csv created successfully!")