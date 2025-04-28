import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Directories
val_dir = '../dataset/validation'
img_size = (256, 256)
batch_size = 32

# Load model
model = load_model('models/densenet_best.h5')

# Validation data generator
val_datagen = ImageDataGenerator(rescale=1./255)
val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Evaluate model
loss, acc = model.evaluate(val_gen, steps=val_gen.samples // batch_size)
print(f'Validation Loss: {loss:.4f}')
print(f'Validation Accuracy: {acc:.4f}')

# Predict and generate metrics
val_gen.reset()
preds = model.predict(val_gen, steps=val_gen.samples // batch_size+1)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes[:len(y_pred)]  # Ensure matching lengths

class_labels = list(val_gen.class_indices.keys())
print('Classification Report:')
print(classification_report(y_true, y_pred, target_names=class_labels))
print('Confusion Matrix:')
print(confusion_matrix(y_true, y_pred))
