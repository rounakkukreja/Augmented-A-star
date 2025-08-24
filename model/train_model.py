import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from .model_utils import load_training_data, prepare_features_labels, create_scaler, save_model

def train_model():
    # Load and prepare data
    data = load_training_data()
    X, y = prepare_features_labels(data)
    
    # Split data (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Feature scaling
    scaler = create_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Initialize and train model
    print("Training Gradient Boosting model...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    
    print("\nModel Evaluation:")
    print(f"Train MAE: {mean_absolute_error(y_train, train_pred):.4f}")
    print(f"Train R²: {r2_score(y_train, train_pred):.4f}")
    print(f"Validation MAE: {mean_absolute_error(y_val, val_pred):.4f}")
    print(f"Validation R²: {r2_score(y_val, val_pred):.4f}")
    
    # Feature importance
    feature_names = ["node_x", "node_y", "goal_x", "goal_y", "manhattan", "euclidean"]
    importances = model.feature_importances_
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importances)
    plt.xlabel("Feature Importance")
    plt.title("Heuristic Model Feature Importance")
    plt.tight_layout()
    plt.savefig("model/feature_importance.png")
    plt.show()
    
    # Save model
    save_model(model, scaler)
    print("Model training complete!")

if __name__ == "__main__":
    train_model()