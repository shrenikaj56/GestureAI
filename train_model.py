import pandas as pd

from src.ml.trainer import GestureTrainer
from src.utils.config import DATASET_PATH


# Load dataset
dataset = pd.read_csv(DATASET_PATH)

print("Dataset:")
print(dataset["gesture"].value_counts())
print()

# Train model
trainer = GestureTrainer()
result = trainer.train(dataset)

# Show results
print("Training completed!")
print(f"Accuracy : {result['accuracy']:.2%}")
print(f"Precision: {result['precision']:.2%}")
print(f"Recall   : {result['recall']:.2%}")
print(f"F1 Score : {result['f1_score']:.2%}")