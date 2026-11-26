# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
Problem Statement: To develop an image classification model using transfer learning
with the pretrained VGG19 convolutional neural network architecture. The pretrained
VGG19 model is adapted by replacing its final classification layer according to the
number of classes in the given dataset.

Dataset: The dataset consists of labeled images belonging to different classes. The
images are divided into training, validation, and testing datasets. The images are resized
to the input size required by VGG19 and normalized before being given to the model.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Load the image dataset and divide it into training, validation, and testing sets. Apply the
required image resizing, normalization, and augmentation.

### STEP 2:

Load the pretrained VGG19 model with weights learned from ImageNet.

### STEP 3:

Freeze the pretrained convolutional layers so that the learned feature representations
are retained.

### STEP 4:

Replace the final fully connected classification layer with a new layer having the same
number of output neurons as the dataset classes.

### STEP 5:

Define the loss function and optimizer, then train the modified model using the training
dataset while monitoring validation performance.

### STEP 6:

Evaluate the trained model using the test dataset and generate the training/validation
loss plot, confusion matrix, classification report, and predictions for new sample images.






## PROGRAM

### Name: KOMALAVARSHINI.S

### Register Number: 212224230133

```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
transform = transforms.Compose([
transforms. Resize((224, 224)), # Resize images for pre-trained model il
transforms. ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # S
dataset_path = "C:\\Users\\admin\\Downloads\\chip_data\\dataset"
train_dataset = datasets. ImageFolder(root=f"{dataset_path}/train", transfori
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=

def show_sample_images(dataset, num_images=5):
fig, axes = plt.subplots(1, num_images, figsize=(5, 5))
for i in range(num_images):
image, label = dataset[i]
image = image.permute(1, 2, 0) # Convert tensor format (C, H, W) ti
axes[i].imshow(image)
axes[i].set_title(dataset.classes[label])
axes[i].axis("off")
plt. show()
show_sample_images(train_dataset)
print(f"Total number of training samples: {len(train_dataset)}")
first_image, label = train_dataset[0]
print(f"Shape of the first image: {first_image. shape}")
print(f"Total number of training samples: {len(test_dataset)}")

first_image1, label = test_dataset[0]
print(f"Shape of the first image : {first_image. shape}")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes=len(train_dataset.classes)
print("Number of classes:", num_classes)
model=models.vgg19(weights = models.VGG19_Weights.DEFAULT)
for param in model.parameters():
param.requires_grad=False
model.classifier[6]=nn.Linear(model.classifier[6].in_features, num_classes)

# Freeze all layers except the final layer
for param in model.features.parameters():
param.requires_grad = False # Freeze feature extractor layers

# Include the Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), 1r=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torchsummary import summary
# Print model summary
summary(model, input_size=(3, 224, 224))

def train_model(model, train_loader,test_loader,num_epochs=10):
train_losses = []
val_losses = []
model.train()
for epoch in range(num_epochs):
running_loss = 0.0
for images, labels in train_loader:
images, labels = images.to(device), labels.to(device)
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
running_loss += loss.item()
train_losses.append(running_loss / len(train_loader))

# Compute validation loss
model.eval()
val_loss = 0.0
with torch.no_grad():
for images, labels in test_loader:
images, labels = images.to(device), labels.to(device)
outputs = model(images)
loss = criterion(outputs, labels)
val_loss += loss.item()

val_losses.append(val_loss / len(test_loader))
model.train()

print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[ -:

plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', ma
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss',
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt. show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

## Step 4: Test the Model and Compute Confusion Matrix & Classification Rep
def test_model(model, test_loader):
model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
for images, labels in test_loader:
images, labels = images.to(device), labels.to(device)
outputs = model(images)
_, predicted = torch.max(outputs, 1)
total += labels.size(0)
correct += (predicted == labels).sum().item()
all_preds.extend(predicted.cpu().numpy())
all_labels.extend(labels.cpu().numpy())

accuracy = correct / total
print(f'Test Accuracy: {accuracy :. 4f}')

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_da
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt. show()

# Print classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=train_d.

train_model(model, train_loader,test_loader)
test_model(model, test_loader)

def predict_image(model, image_index, dataset):
model.eval()
image, label = dataset[image_index]
with torch.no_grad():
image_tensor = image.unsqueeze(0).to(device)
output = model(image_tensor)

# Apply sigmoid to get probability, threshold at 0.5
prob = torch.sigmoid(output)
output2=torch.max(prob,1)
predicted = output2[1].item()

class_names = class_names = dataset.classes
# Display the image
image_to_display = transforms. ToPILImage() (image)
plt.figure(figsize=(4, 4))
plt. imshow(image_to_display)
plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predi
plt.axis("off")
plt.show()
print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted

# Example Prediction
predict_image(model, image_index=25, dataset=test_dataset)
```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="782" height="596" alt="image" src="https://github.com/user-attachments/assets/d2df9744-6720-4a74-8be4-68dabee4859d" />

## Confusion Matrix

<img width="833" height="780" alt="image" src="https://github.com/user-attachments/assets/a008e1dd-cbdc-4b39-81dc-48f57bc49b3e" />

## Classification Report
<img width="827" height="313" alt="image" src="https://github.com/user-attachments/assets/ab1e8851-a7b4-41c4-a55a-8a479f3f4d81" />

### New Sample Data Prediction
<img width="692" height="595" alt="image" src="https://github.com/user-attachments/assets/d568c23c-4c98-4fc4-9cf2-f83bbdf51627" />

<img width="695" height="618" alt="image" src="https://github.com/user-attachments/assets/afb722f2-6ca8-40ac-a29c-9b76e29060d6" />


## RESULT
VGG19 model was fine-tuned and tested successfully. The model achieved good accuracy with correct predictions on sample test images.
