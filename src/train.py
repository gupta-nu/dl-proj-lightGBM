import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from src.preprocessing import load_and_preprocess
from src.model import train_lgb
import joblib

def train_and_evaluate():
    # Load and preprocess the dataset
    df = load_and_preprocess("data/train.csv")

    # Features for training
    features = ['condition', 'color_type', 'length(m)', 'height(cm)', 'X1', 'X2']
    X = df[features]

    # Targets
    y_breed = df['breed_category']
    y_pet = df['pet_category']

    # Split data once for both targets to keep train/val indices aligned
    X_train, X_val, y_breed_train, y_breed_val, y_pet_train, y_pet_val = train_test_split(
        X, y_breed, y_pet, test_size=0.2, random_state=42
    )

    print("[INFO] Training breed_category model...")
    breed_model = train_lgb(X_train, y_breed_train)

    print("[INFO] Training pet_category model...")
    pet_model = train_lgb(X_train, y_pet_train)

    # Predict on validation set
    y_breed_pred = breed_model.predict(X_val)
    y_pet_pred = pet_model.predict(X_val)

    # Print classification reports
    print("Breed Classification Report:")
    print(classification_report(y_breed_val, y_breed_pred))
    print("Pet Classification Report:")
    print(classification_report(y_pet_val, y_pet_pred))

    # Plot confusion matrices side by side
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(confusion_matrix(y_breed_val, y_breed_pred), annot=True, ax=ax[0], cmap='Blues', fmt='d')
    ax[0].set_title("Breed Category Confusion Matrix")
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")

    sns.heatmap(confusion_matrix(y_pet_val, y_pet_pred), annot=True, ax=ax[1], cmap='Greens', fmt='d')
    ax[1].set_title("Pet Category Confusion Matrix")
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("True")

    plt.tight_layout()

    # Ensure output directory exists
    save_dir = "outputs/visualizations"
    os.makedirs(save_dir, exist_ok=True)

    # Save confusion matrices figure
    plt.savefig(os.path.join(save_dir, "confusion_matrices.png"))
    plt.close()

    # Save trained models
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(breed_model, "outputs/breed_model.pkl")
    joblib.dump(pet_model, "outputs/pet_model.pkl")

    print(f"[INFO] Models and confusion matrices saved in 'outputs/' directory.")

    return breed_model, pet_model

if __name__ == "__main__":
    train_and_evaluate()
