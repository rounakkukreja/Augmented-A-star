import json
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_training_data(file_path="data/train_paths.json"):
    """Load and parse training dataset"""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def prepare_features_labels(data):
    """Convert raw data to feature matrix and target vector"""
    features = []
    labels = []
    
    for sample in data:
        node_x, node_y = sample["node"]
        goal_x, goal_y = sample["goal"]
        
        # Feature vector: [node_x, node_y, goal_x, goal_y, manhattan, euclidean]
        features.append([
            node_x,
            node_y,
            goal_x,
            goal_y,
            sample["manhattan"],
            sample["euclidean"]
        ])
        
        # Target: actual optimal cost (h_star)
        labels.append(sample["h_star"])
    
    return np.array(features), np.array(labels)

def create_scaler(features):
    """Create and fit feature scaler"""
    scaler = StandardScaler()
    scaler.fit(features)
    return scaler

def save_model(model, scaler, model_path="model/heuristic_model.pkl"):
    """Save trained model and scaler"""
    from joblib import dump
    dump({"model": model, "scaler": scaler}, model_path)
    print(f"Saved model to {model_path}")

def load_model(model_path="model/heuristic_model.pkl"):
    """Load trained model and scaler"""
    from joblib import load
    data = load(model_path)
    return data["model"], data["scaler"]