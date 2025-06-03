import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

#Function to load images from a custom binary file
def load_images(path):

  with open(path, 'rb') as f: 
    _ = int.from_bytes(f.read(4), 'big') #Magic Number, typically ignored
    n_imgs = int.from_bytes(f.read(4), 'big') # Number of images
    n_rows = int.from_bytes(f.read(4), 'big') # Image height
    n_cols = int.from_bytes(f.read(4), 'big') # Image Width
    buf = f.read(n_imgs * n_rows * n_cols) #Image data buffer
  data = np.frombuffer(buf, dtype=np.uint8)
  return data.reshape(n_imgs, n_rows, n_cols) # Reshape buffer to image array
  
#Function to load labels from a binary file
def load_labels(path):
  with open(path, 'rb') as f:
    _ = int.from_bytes(f.read(4), 'big') #Magic number
    n_labels = int.from_bytes(f.read(4), 'big') # Number of labels
    buf = f.read(n_labels) # Label Data buffer
  return np.frombuffer(buf, dtype=np.uint8)

#Load training and testing images/labels
test_images = load_images('/content/t10k-images-idx3-ubyte')
test_labels = load_labels('/content/t10k-labels-idx1-ubyte')
train_images = load_images('/content/train-images-idx3-ubyte ')
train_labels = load_labels('/content/train-labels-idx1-ubyte')

#Output shapes for verification
print(f"test images: {test_images.shape}, Test labels: {test_labels.shape}")
print(f"Train images: {train_images.shape}, Train labels: {train_labels.shape}")

#Visualise a sample test image
plt.imshow(test_images[0], cmap="gray")
plt.title(f"Label = {test_labels[0]}")
plt.axis('off')
plt.show()

#Prepare Data: flatten image and normalise pixel values 
X = np.vstack([train_images, test_images]).reshape(-1, 28*28) / 255.0
y = np.hstack([train_labels, test_labels])


#Split data into training and holdout (validation) sets
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=10000, random_state=42, stratify=y)

#Baseline pipeline: StandardScaler + PCA (95% variance + KNN
baseline_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95)),
    ("knn", KNeighborsClassifier(n_neighbors=3, metric='euclidean'))
])

#Fit a baseline model and evaluate
baseline_pipe.fit(X_train, y_train)
y_pred_base = baseline_pipe.predict(X_holdout)
accuracy_base = accuracy_score(y_holdout, y_pred_base)
print(f"Baseline accuracy: {accuracy_base}")


#Define a pipeline with parameter tuning for PCA components and KNN hyperparameters
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA()), #Number of components to be tuned
    ("knn", KNeighborsClassifier()) #Hyperparameters to be tuned
])

#Define grid search parameters
param_grid = {
    "pca__n_components": [0.90, 0.95], #Variance explained threshold
    "knn__n_neighbors": [3, 5], # Number of neighbours
    "knn__metric": ["euclidean"] #Distance Metrics
}

Perform grid search with cross-validation
grid = GridSearchCV(
    pipe,
    param_grid,
    cv=3,
    n_jobs=-1,
    verbose=2
)
# Fit grid search on training data
grid.fit(X_train, y_train)

#Output best parameters and cross-validation score
print(f"Best parameters: {grid.best_params_}")
print(f"CV accuracy: {grid.best_score_}")

#Evaluate best model on holdout set
best_model = grid.best_estimator_
y_pred = best_model.predict(X_holdout)
accuracy = accuracy_score(y_holdout, y_pred)
print(f"Holdout accuracy: {accuracy}")

#Plot cumulative explaiend variance to determined number of PCs to visualise how many principal components are needed 
pca = best_model.named_steps["pca"]
cumvar = np.cumsum(pca.explained_variance_ratio_) #
plt.figure()
plt.plot(cumvar, lw=2)
plt.axhline(grid.best_params_["pca__n_components"], color="r", ls="--") # Highlight selected number of PCs
plt.xlabel("Number of PCs")
plt.ylabel("Cumulative explained variance")
plt.title("Explained Variance vs. # of Principal Components")
plt.show()

# Generate confusion matrix for predictions 
cm = confusion_matrix(y_holdout, y_pred, labels=grid.classes_)
disp = ConfusionMatrixDisplay(cm, display_labels=grid.classes_)
disp.plot(cmap="Blues", xticks_rotation="vertical") # Plot confusion matrix # Plot Confusion
plt.title("Confusion Matrix")
plt.show()

# Identify misclassified images 
mis_idx = np.where(y_pred != y_holdout)[0]
plt.figure(figsize=(8, 4))
for i, idx in enumerate(mis_idx[:8]): # Show first 8 misclassified images 
  plt.subplot(2, 4, i+1)
  plt.imshow((X_holdout[idx]*255).reshape(28, 28), cmap="gray") # Reshape flattened image
  plt.title(f"true={y_holdout[idx]}, pred={y_pred[idx]}")
  plt.axis("off")
plt.suptitle("Misclassified Images")
plt.tight_layout()
plt.show()




