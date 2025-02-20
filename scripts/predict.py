import pickle
from sklearn.datasets import load_iris

# Load model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load Iris dataset
iris = load_iris()

# Example prediction (using the first sample from the dataset)
example = iris.data[0].reshape(1, -1)
prediction = model.predict(example)

# Map prediction to class name
predicted_class = iris.target_names[prediction][0]
print(f"Prediction: {predicted_class}")
