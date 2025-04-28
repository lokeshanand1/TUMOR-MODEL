# import pickle
# import matplotlib.pyplot as plt
# import os

# def plot_training_curves(history, optimizer_name='Adam'):
#     epochs = range(1, len(history['accuracy']) + 1)

#     plt.figure(figsize=(14, 6))

#     # Plot Accuracy
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, history['accuracy'], 'b-', linewidth=2, label='Training Accuracy')
#     plt.plot(epochs, history['val_accuracy'], 'r--', linewidth=2, label='Validation Accuracy')
#     plt.title(f'{optimizer_name} Optimizer - Accuracy per Epoch', fontsize=14)
#     plt.xlabel('Epoch', fontsize=12)
#     plt.ylabel('Accuracy', fontsize=12)
#     plt.legend()
#     plt.grid(True)

#     # Plot Loss
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, history['loss'], 'g-', linewidth=2, label='Training Loss')
#     plt.plot(epochs, history['val_loss'], 'm--', linewidth=2, label='Validation Loss')
#     plt.title(f'{optimizer_name} Optimizer - Loss per Epoch', fontsize=14)
#     plt.xlabel('Epoch', fontsize=12)
#     plt.ylabel('Loss', fontsize=12)
#     plt.legend()
#     plt.grid(True)

#     plt.tight_layout()
#     os.makedirs('results/training_curves', exist_ok=True)
#     plt.savefig(f'results/training_curves/{optimizer_name.lower()}_training_curves.png', dpi=300)
#     plt.show()

# def main():
#     # Load training history saved during training
#     history_path = 'models/training_history.pkl'
#     if not os.path.exists(history_path):
#         print(f"Training history file not found at {history_path}. Please run training first.")
#         return

#     with open(history_path, 'rb') as f:
#         history = pickle.load(f)

#     # Plot graphs for Adam optimizer (assuming training used Adam)
#     plot_training_curves(history, optimizer_name='Adam')

#     # If you have training histories for other optimizers (e.g., RMSprop),
#     # load and plot them similarly by repeating the above steps.

# if __name__ == '__main__':
#     main()


# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.applications import DenseNet121
# from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam, RMSprop, SGD, Adagrad
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # Paths to preprocessed data
# train_dir = '../dataset/train'
# val_dir = '../dataset/validation'
# img_size = (256, 256)

# # Hyperparameter grids to test
# learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
# optimizers_dict = {
#     'Adam': Adam,
#     'RMSprop': RMSprop,
#     'SGD': SGD,
#     'Adagrad': Adagrad
# }
# batch_sizes = [4, 8, 10, 16, 20]

# # Fixed parameters
# num_classes = 3
# epochs = 25  

# # Function to create model
# def create_densenet():
#     base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=img_size + (3,))
#     base_model.trainable = False
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
#     output = Dense(num_classes, activation='softmax')(x)
#     model = Model(inputs=base_model.input, outputs=output)
#     return model

# # Function to get optimizer instance
# def get_optimizer(name, lr):
#     return optimizers_dict[name](learning_rate=lr)

# # Store results
# results_lr = []
# results_opt = []
# results_bs = []

# # For epoch vs error plot of last run
# last_history = None

# print("Starting experiments...\n")

# # 1. Vary initial learning rate (with fixed optimizer and batch size)
# fixed_optimizer = 'Adam'
# fixed_batch_size = 32
# print("Testing different learning rates with Adam optimizer and batch size 32:")
# for lr in learning_rates:
#     print(f"Training with learning rate = {lr}")
#     model = create_densenet()
#     optimizer = get_optimizer(fixed_optimizer, lr)
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#     train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         rotation_range=15,
#         width_shift_range=0.1,
#         height_shift_range=0.1,
#         zoom_range=0.1,
#         horizontal_flip=True
#     )
#     val_datagen = ImageDataGenerator(rescale=1./255)

#     train_gen = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=fixed_batch_size, class_mode='categorical')
#     val_gen = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=fixed_batch_size, class_mode='categorical')

#     history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, steps_per_epoch=train_gen.samples // fixed_batch_size, validation_steps=val_gen.samples // fixed_batch_size, verbose=0)

#     best_val_acc = max(history.history['val_accuracy'])
#     results_lr.append((lr, best_val_acc))
#     print(f"  Best val accuracy: {best_val_acc:.4f}")

# # 2. Vary optimizer (with fixed learning rate and batch size)
# fixed_lr = 1e-3
# print("\nTesting different optimizers with learning rate 0.001 and batch size 32:")
# for opt_name in optimizers_dict.keys():
#     print(f"Training with optimizer = {opt_name}")
#     model = create_densenet()
#     optimizer = get_optimizer(opt_name, fixed_lr)
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#     train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         rotation_range=15,
#         width_shift_range=0.1,
#         height_shift_range=0.1,
#         zoom_range=0.1,
#         horizontal_flip=True
#     )
#     val_datagen = ImageDataGenerator(rescale=1./255)

#     train_gen = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=fixed_batch_size, class_mode='categorical')
#     val_gen = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=fixed_batch_size, class_mode='categorical')

#     history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, steps_per_epoch=train_gen.samples // fixed_batch_size, validation_steps=val_gen.samples // fixed_batch_size, verbose=0)

#     best_val_acc = max(history.history['val_accuracy'])
#     results_opt.append((opt_name, best_val_acc))
#     print(f"  Best val accuracy: {best_val_acc:.4f}")

# # 3. Vary batch size (with fixed learning rate and optimizer)
# fixed_optimizer = 'Adam'
# fixed_lr = 1e-3
# print("\nTesting different batch sizes with Adam optimizer and learning rate 0.001:")
# for bs in batch_sizes:
#     print(f"Training with batch size = {bs}")
#     model = create_densenet()
#     optimizer = get_optimizer(fixed_optimizer, fixed_lr)
#     model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

#     train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         rotation_range=15,
#         width_shift_range=0.1,
#         height_shift_range=0.1,
#         zoom_range=0.1,
#         horizontal_flip=True
#     )
#     val_datagen = ImageDataGenerator(rescale=1./255)

#     train_gen = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=bs, class_mode='categorical')
#     val_gen = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=bs, class_mode='categorical')

#     history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, steps_per_epoch=train_gen.samples // bs, validation_steps=val_gen.samples // bs, verbose=0)

#     best_val_acc = max(history.history['val_accuracy'])
#     results_bs.append((bs, best_val_acc))
#     print(f"  Best val accuracy: {best_val_acc:.4f}")

#     # Save last history for epoch vs error plot
#     last_history = history

# print("\nAll experiments completed.\n")

# # --- Plotting all graphs ---

# # 1. Epoch vs Error (last run)
# if last_history is not None:
#     epochs_range = range(1, len(last_history.history['val_accuracy']) + 1)
#     train_error = [1 - acc for acc in last_history.history['accuracy']]
#     val_error = [1 - acc for acc in last_history.history['val_accuracy']]

#     plt.figure(figsize=(8,6))
#     plt.plot(epochs_range, train_error, 'b-', label='Training Error')
#     plt.plot(epochs_range, val_error, 'r--', label='Validation Error')
#     plt.title('Epoch vs Error (Last Run)')
#     plt.xlabel('Epoch')
#     plt.ylabel('Error')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
# else:
#     print("No training history available for epoch vs error plot.")

# # 2. Initial Learning Rate vs Accuracy
# lrs, lr_accs = zip(*results_lr)
# plt.figure(figsize=(8,6))
# plt.plot(lrs, lr_accs, 'o-', linewidth=2)
# plt.xscale('log')
# plt.title('Initial Learning Rate vs Validation Accuracy')
# plt.xlabel('Initial Learning Rate (log scale)')
# plt.ylabel('Validation Accuracy')
# plt.grid(True)
# plt.show()

# # 3. Optimizer vs Accuracy
# opts, opt_accs = zip(*results_opt)
# plt.figure(figsize=(8,6))
# plt.bar(opts, opt_accs, color=['blue', 'orange', 'green', 'red'])
# plt.title('Optimizer vs Validation Accuracy')
# plt.xlabel('Optimizer')
# plt.ylabel('Validation Accuracy')
# plt.ylim(0,1)
# plt.grid(axis='y')
# plt.show()

# # 4. Batch Size vs Accuracy
# bss, bs_accs = zip(*results_bs)
# plt.figure(figsize=(8,6))
# plt.plot(bss, bs_accs, 's-', linewidth=2, color='purple')
# plt.title('Batch Size vs Validation Accuracy')
# plt.xlabel('Batch Size')
# plt.ylabel('Validation Accuracy')
# plt.grid(True)
# plt.show()

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Paths
summary_csv = 'results/experiment_summary.csv'
histories_dir = 'results/histories'

# Load experiment summary CSV
if not os.path.exists(summary_csv):
    raise FileNotFoundError(f"Experiment summary CSV not found at {summary_csv}. Run training first.")

df = pd.read_csv(summary_csv)

# --- 1. Epoch vs Error graph ---

# Use the last experiment's history for epoch vs error plot
last_history_path = df.iloc[-1]['history_path']
if not os.path.exists(last_history_path):
    raise FileNotFoundError(f"History file not found: {last_history_path}")

with open(last_history_path, 'rb') as f:
    history = pickle.load(f)

epochs = range(1, len(history['val_accuracy']) + 1)
train_error = [1 - acc for acc in history['accuracy']]
val_error = [1 - acc for acc in history['val_accuracy']]

plt.figure(figsize=(8,6))
plt.plot(epochs, train_error, 'b-', label='Training Error')
plt.plot(epochs, val_error, 'r--', label='Validation Error')
plt.title('Epoch vs Error (Last Experiment)')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- 2. Initial Learning Rate vs Validation Accuracy ---

lr_df = df[df['experiment_type'] == 'learning_rate'].copy()
if not lr_df.empty:
    lr_df.sort_values(by='value', inplace=True)
    plt.figure(figsize=(8,6))
    plt.plot(lr_df['value'], lr_df['best_val_accuracy'], 'o-', linewidth=2)
    plt.xscale('log')
    plt.title('Initial Learning Rate vs Validation Accuracy')
    plt.xlabel('Initial Learning Rate (log scale)')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No learning rate experiment data found.")

# --- 3. Optimizer vs Validation Accuracy ---

opt_df = df[df['experiment_type'] == 'optimizer'].copy()
if not opt_df.empty:
    # Sort by accuracy descending for better visualization
    opt_df.sort_values(by='best_val_accuracy', ascending=False, inplace=True)
    plt.figure(figsize=(8,6))
    plt.bar(opt_df['value'], opt_df['best_val_accuracy'], color='skyblue')
    plt.title('Optimizer vs Validation Accuracy')
    plt.xlabel('Optimizer')
    plt.ylabel('Validation Accuracy')
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
else:
    print("No optimizer experiment data found.")

# --- 4. Batch Size vs Validation Accuracy ---

bs_df = df[df['experiment_type'] == 'batch_size'].copy()
if not bs_df.empty:
    bs_df.sort_values(by='value', inplace=True)
    plt.figure(figsize=(8,6))
    plt.plot(bs_df['value'], bs_df['best_val_accuracy'], 's-', linewidth=2, color='purple')
    plt.title('Batch Size vs Validation Accuracy')
    plt.xlabel('Batch Size')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No batch size experiment data found.")





