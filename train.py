import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from utils import load_data, split_data, create_model

X, y = load_data()
data = split_data(X, y, test_size=0.1, valid_size=0.1)
model = create_model()

tensorboard = TensorBoard(log_dir="logs")
early_stopping = EarlyStopping(mode="min", patience=40, restore_best_weights=True)

batch_size = 64
epochs = 200

model.fit(data["X_train"], data["y_train"],
          epochs=epochs, batch_size=batch_size,
          validation_data=(data["X_valid"], data["y_valid"]),
          callbacks=[tensorboard, early_stopping])

model.save("results/model.h5")

print(f"Evaluating the model using {len(data['X_test'])} samples...")
loss, accuracy = model.evaluate(data["X_test"], data["y_test"], verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy*100:.2f}%")
