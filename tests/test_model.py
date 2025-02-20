import pickle
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    # Load model
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load Iris dataset
    iris = load_iris()
    X_test = iris.data
    y_test = iris.target

    # Predict
    y_pred = model.predict(X_test)

    # Check accuracy
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy >= 0.9, "Model accuracy is too low"
