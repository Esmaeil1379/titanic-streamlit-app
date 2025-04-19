from sklearn.ensemble import RandomForestClassifier
import joblib

# Create and save a simple model
model = RandomForestClassifier()
joblib.dump(model, 'test_model.pkl')

# Load and print it
loaded_model = joblib.load('test_model.pkl')
print("Model loaded successfully:", loaded_model)
