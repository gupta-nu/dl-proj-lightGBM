import pandas as pd
import joblib
from src.preprocessing import load_and_preprocess

def predict_and_save_submission():
    test_df = load_and_preprocess("data/test.csv")
    test_features = ['condition', 'color_type', 'length(m)', 'height(cm)', 'X1', 'X2']
    
    breed_model = joblib.load("outputs/breed_model.pkl")
    pet_model = joblib.load("outputs/pet_model.pkl")
    
    breed_preds = breed_model.predict(test_df[test_features])
    pet_preds = pet_model.predict(test_df[test_features])
    
    submission = pd.DataFrame({
        "pet_id": test_df["pet_id"],
        "breed_category": breed_preds,
        "pet_category": pet_preds
    })
    
    submission.to_csv("outputs/submission.csv", index=False)
    print("[INFO] Submission saved to outputs/submission.csv")