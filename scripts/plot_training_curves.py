import matplotlib.pyplot as plt

def plot_training_curves(history, optimizer_name):
    plt.figure(figsize=(15, 6))
    
    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], 'b-', linewidth=2, label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], 'r--', linewidth=2, label='Validation Accuracy')
    plt.title(f'{optimizer_name} - Accuracy Curves', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend()
    
    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], 'g-', linewidth=2, label='Train Loss')
    plt.plot(history.history['val_loss'], 'm--', linewidth=2, label='Validation Loss')
    plt.title(f'{optimizer_name} - Loss Curves', fontsize=14)
    plt.ylabel('Loss', fontsize=12)
    plt.xlabel('Epoch', fontsize=12)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Example usage for Adam optimizer (as per paper parameters)
history_adam = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // 32,  # Batch size 32 from paper
    epochs=50,  # From paper specifications
    validation_data=val_generator,
    validation_steps=val_generator.n // 32
)
plot_training_curves(history_adam, 'Adam')

# For RMSprop comparison (as mentioned in paper)
model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_rmsprop = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // 32,
    epochs=50,
    validation_data=val_generator,
    validation_steps=val_generator.n // 32
)
plot_training_curves(history_rmsprop, 'RMSprop')
