from src.train import train_and_evaluate
from src.inference import predict_and_save_submission

if __name__ == "__main__":
    print("[INFO] Training models...")
    train_and_evaluate()
    
    print("[INFO] Generating predictions...")
    predict_and_save_submission()