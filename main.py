# main.py

import pandas as pd
import joblib

# Load the saved model
model = joblib.load('titanic_model.pkl')

# Example input data for prediction (adjust as needed)
# Columns: ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
sample_data = pd.DataFrame([{
    'Pclass': 3,
    'Sex': 1,     # 1 for male, 0 for female (assuming you used label encoding like in your earlier script)
    'Age': 22.0,
    'Fare': 7.25
}])


# Make prediction
prediction = model.predict(sample_data)

print("Prediction:", "Survived" if prediction[0] == 1 else "Did not survive")
