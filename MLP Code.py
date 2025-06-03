import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix

#Function to load images from binary format
def load_images(path):
  with open(path, 'rb') as f:
    _ = int.from_bytes(f.read(4), 'big') # Magic number
    n_imgs = int.from_bytes(f.read(4), 'big') # Number of images
    n_rows = int.from_bytes(f.read(4), 'big') # Image height (28)
    n_cols = int.from_bytes(f.read(4), 'big') # Image width (28)
    buf = f.read(n_imgs * n_rows * n_cols) # Read image data
  data = np.frombuffer(buf, dtype=np.uint8)
  return data.reshape(n_imgs, n_rows, n_cols) 

# Function to load labels from binary format  
def load_labels(path):
  with open(path, 'rb') as f:
    _ = int.from_bytes(f.read(4), 'big') # Magic number
    n_labels = int.from_bytes(f.read(4), 'big') # Number of labels
    buf = f.read(n_labels) # Read label data
  return np.frombuffer(buf, dtype=np.uint8)

# Load MNIST dataset
test_images = load_images('/content/t10k-images-idx3-ubyte')
test_labels = load_labels('/content/t10k-labels-idx1-ubyte')
train_images = load_images('/content/train-images-idx3-ubyte ')
train_labels = load_labels('/content/train-labels-idx1-ubyte')

# Verify data shapes
print(f"test images: {test_images.shape}, Test labels: {test_labels.shape}")
print(f"Train images: {train_images.shape}, Train labels: {train_labels.shape}")

# Display sample image 
plt.imshow(test_images[0], cmap='gray')
plt.title(f"Label: {test_labels[0]}")
plt.axis('off')
plt.show()

# Data preprocessing: flatten and normalise pixel values to [0, 1]
X = np.vstack([train_images, test_images]).reshape(-1, 28*28) / 255.0
y = np.hstack([train_labels, test_labels])

# Split data into training and holdout sets 
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=10000, random_state=42, stratify=y)

# Feature scaling: standardise features to have mean=0, std=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_holdout_scaled = scaler.transform(X_holdout)

# Define hyperparameter search space
learning_rates = [1e-3, 1e-4]
alphas = [1e-3, 1e-4]
activations = ['relu', 'logistic', 'tanh']

results = []

# Grid search over hyperparameters 
for lr in learning_rates:
  for alpha in alphas:
    for activation in activations:
      # Create MLP with specific architecture and hyperparameters
      mlp = MLPClassifier(hidden_layer_sizes=(128, 64), # two hidden layers: 128, 64
                    activation=activation, # Activation function 
                    solver='adam', #Optimisation algorithm
                    alpha=alpha, # L2 regularisation 
                    learning_rate_init=lr, # Initial learning rate
                    max_iter=50, # Max iterations
                    batch_size=256, # Number of samples per gradient
                    random_state=42, # For reproducibility 
                    verbose=True, # Print Prgress messages
                    early_stopping=True, # Stop training when validation score stops improving
                    validation_fraction=0.1 # Fraction of training data used for early stopping
                
                    )
      mlp.fit(X_train_scaled, y_train) # Train model
      validation_score = mlp.score(X_holdout_scaled, y_holdout) # Evaluate on holdout set
      results.append((lr, alpha, activation, validation_score)) # Store results for comparison
      print(f"Learning Rate: {lr}, Alpha: {alpha}, Activation: {activation}, Validation Score: {validation_score:.4f}") # Store results for comparison
      
best_result = max(results, key=lambda x: x[3]) # Find best parameter combination
print(f"Best Results: Learning Rate: {best_result[0]}, Alpha: {best_result[1]}, Activation: {best_result[2]}, Validation Score: {best_result[3]:.4f}")

# Train final model with best hyperparameters
best_mlp = MLPClassifier(hidden_layer_sizes=(128, 64),
                         activation=best_result[2],
                         solver='adam',
                         alpha=best_result[1],
                         learning_rate_init=best_result[0],
                         max_iter=50,
                         batch_size=256,
                         random_state=42,
                         verbose=True,
                         early_stopping=True,
                         validation_fraction=0.1
                         )
best_mlp.fit(X_train_scaled, y_train) # Fit the best model and make predictions
y_pred_mlp = best_mlp.predict(X_holdout_scaled)

# Plot training curves to visualise model performance during training
plt.figure(figsize=(8, 4))
# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(best_mlp.loss_curve_, label='Training Loss', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('MLP Training Loss Curve')
plt.grid(True)
# Plot validation accuracy curve
plt.subplot(1, 2, 2)
plt.plot(best_mlp.validation_scores_, label='Validation Accuracy',linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('MLP Validation Accuracy Curve')
plt.grid(True)

plt.tight_layout()
plt.show()

# Generate confusion matrix for detailed performance analysis
cm = confusion_matrix(y_holdout, y_pred_mlp, labels=range(10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(10))

plt.figure(figsize=(6, 6))
disp.plot(cmap='Blues', xticks_rotation='vertical', ax=plt.gca())
plt.title('MLP Confusion Matrix on Hold-out')
plt.show()

# Identify and visualise missclassified examples
mis = np.where(y_pred_mlp != y_holdout)[0]

plt.figure(figsize=(8, 4))
for i, idx in enumerate(mis[:8]): # Show first 8 missclassified images
  plt.subplot(2, 4, i+1)
  plt.imshow(X_holdout[idx].reshape(28, 28), cmap='gray')
  plt.title(f"Predicted: {y_holdout[idx]}, pred={y_pred_mlp[idx]}")
  plt.axis('off')
plt.suptitle("MLP Misclassifications (First 8)")
plt.tight_layout()
plt.show()
