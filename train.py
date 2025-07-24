from data_loader import load_dataset
from model_builder import build_model

train_data, test_data, info = load_dataset()
num_classes = info.features['label'].num_classes

model = build_model(num_classes)
model.fit(train_data, epochs=5, validation_data=test_data)

model.save('cat_breed_model.h5')
print("✅ Model saved as cat_breed_model.h5")
with open("class_names.txt", "w") as f:
    for name in info.features['label'].names:
        f.write(f"{name}\n")
