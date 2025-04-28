import os
import csv
import numpy as np
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pickle

# Paths to preprocessed data
train_dir = '../dataset/train'
val_dir = '../dataset/validation'
img_size = (256, 256)

# Hyperparameter grids
learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
optimizers_dict = {
    'Adam': Adam,
    'RMSprop': RMSprop,
    'SGD': SGD,
    'Adagrad': Adagrad
}
batch_sizes = [4, 8, 10, 16, 20]

# Fixed parameters
num_classes = 3
epochs = 50  # Use 50 epochs as per paper

# Create results directories
os.makedirs('results/histories', exist_ok=True)
summary_csv = 'results/experiment_summary.csv'

# Initialize CSV file with header if it doesn't exist
if not os.path.exists(summary_csv):
    with open(summary_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['experiment_type', 'parameter', 'value', 'best_val_accuracy', 'history_path'])

# Model creation function
def create_densenet():
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=img_size + (3,))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    return model

# Get optimizer instance
def get_optimizer(name, lr):
    return optimizers_dict[name](learning_rate=lr)

# Save training history to pickle
def save_history(history, filename):
    with open(filename, 'wb') as f:
        pickle.dump(history.history, f)

# Append experiment result to CSV
def append_to_csv(experiment_type, parameter, value, best_val_acc, history_path):
    with open(summary_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([experiment_type, parameter, value, best_val_acc, history_path])

# Data generators
def get_data_generators(batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
    val_gen = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
    return train_gen, val_gen

# Training loop for experiments
def run_experiments():
    print("Starting experiments...\n")

    # 1. Vary initial learning rate (fixed optimizer=Adam, batch_size=32)
    fixed_optimizer = 'Adam'
    fixed_batch_size = 32
    print("Experiment: Varying Learning Rate")
    for lr in learning_rates:
        print(f"Training with learning rate = {lr}")
        model = create_densenet()
        optimizer = get_optimizer(fixed_optimizer, lr)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        train_gen, val_gen = get_data_generators(fixed_batch_size)

        history = model.fit(train_gen, epochs=epochs, validation_data=val_gen,
                            steps_per_epoch=train_gen.samples // fixed_batch_size,
                            validation_steps=val_gen.samples // fixed_batch_size,
                            verbose=1)

        best_val_acc = max(history.history['val_accuracy'])
        history_file = f'results/histories/history_lr_{lr}.pkl'
        save_history(history, history_file)
        append_to_csv('learning_rate', 'lr', lr, best_val_acc, history_file)
        print(f"  Best val accuracy: {best_val_acc:.4f}\n")

    # 2. Vary optimizer (fixed lr=0.001, batch_size=32)
    fixed_lr = 1e-3
    print("Experiment: Varying Optimizer")
    for opt_name in optimizers_dict.keys():
        print(f"Training with optimizer = {opt_name}")
        model = create_densenet()
        optimizer = get_optimizer(opt_name, fixed_lr)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        train_gen, val_gen = get_data_generators(fixed_batch_size)

        history = model.fit(train_gen, epochs=epochs, validation_data=val_gen,
                            steps_per_epoch=train_gen.samples // fixed_batch_size,
                            validation_steps=val_gen.samples // fixed_batch_size,
                            verbose=1)

        best_val_acc = max(history.history['val_accuracy'])
        history_file = f'results/histories/history_opt_{opt_name}.pkl'
        save_history(history, history_file)
        append_to_csv('optimizer', 'opt', opt_name, best_val_acc, history_file)
        print(f"  Best val accuracy: {best_val_acc:.4f}\n")

    # 3. Vary batch size (fixed optimizer=Adam, lr=0.001)
    print("Experiment: Varying Batch Size")
    fixed_optimizer = 'Adam'
    fixed_lr = 1e-3
    for bs in batch_sizes:
        print(f"Training with batch size = {bs}")
        model = create_densenet()
        optimizer = get_optimizer(fixed_optimizer, fixed_lr)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        train_gen, val_gen = get_data_generators(bs)

        history = model.fit(train_gen, epochs=epochs, validation_data=val_gen,
                            steps_per_epoch=train_gen.samples // bs,
                            validation_steps=val_gen.samples // bs,
                            verbose=1)

        best_val_acc = max(history.history['val_accuracy'])
        history_file = f'results/histories/history_bs_{bs}.pkl'
        save_history(history, history_file)
        append_to_csv('batch_size', 'bs', bs, best_val_acc, history_file)
        print(f"  Best val accuracy: {best_val_acc:.4f}\n")

    print("All experiments completed.")

if __name__ == '__main__':
    run_experiments()
