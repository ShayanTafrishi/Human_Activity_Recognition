

# %%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import torch
import torchvision
import torchvision.transforms as transforms
import zipfile





# %%
df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')

print("The shape of the train dataset is:", df_train.shape )
print("The shape of the test dataset is:", df_test.shape )


# %%
# Check for any missing values in the training set
print("Amount of missing values in train",df_train.isnull().sum().sum())
print("Amount of missing values in test",df_test.isnull().sum().sum())


# %%

# Check the balance of the target class
activities = df_train['Activity'].value_counts()

plt.figure(figsize=(10,10))
plt.pie(activities, labels=activities.index, autopct='%1.1f%%', startangle=90)
plt.title('Balance of different activity types')


plt.show()

# %%
# Extract the feature names
feature_names = df_train.columns[:-1] # Exclude the 'Activity' column

# Count 'Acc' and 'Gyro' occurrences in feature names
acc_count = sum('Acc' in name for name in feature_names)
gyro_count = sum('Gyro' in name for name in feature_names)
other_count = len(feature_names) - acc_count - gyro_count

# Plot the pie chart
counts = [acc_count, gyro_count, other_count]
labels = ['Accelerometer', 'Gyroscope', 'Other']

plt.figure(figsize=(10,10))
plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Sensor type usage in features')
plt.show()


# %% [markdown]
# # Step 2: [Optional] Select the variables to keep
# 
# As per the instructions, variable selection is optional. Since it's not clear from the dataset description whether some features are more important than others, we'll not perform any feature selection for now. If, during model evaluation, we find that our model isn't performing well, we may revisit this step.

# %% [markdown]
# # Step 3: Pre-process the dataset

# %%

from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader

# Convert the activity labels to integers
le = LabelEncoder()
df_train['Activity'] = le.fit_transform(df_train['Activity'])
df_test['Activity'] = le.transform(df_test['Activity'])

# Get features and labels
X_train = df_train.iloc[:, :-1].values
y_train = df_train.iloc[:, -1].values

X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.long)

X_test = torch.tensor(X_test, dtype=torch.float)
y_test = torch.tensor(y_test, dtype=torch.long)



#X_train=df_train.iloc[:,:-2]
#y_train=df_train.iloc[:,-1]

#X_test=df_test.iloc[:,:-2]
#y_test=df_test.iloc[:,-1]

# %%
class HARdataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


# %%
# Create datasets
train_dataset = HARdataset(X_train, y_train)
test_dataset = HARdataset(X_test, y_test)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)



# %% [markdown]
# # Step 5: Create a neural architecture in Pytorch
# 

# %%
from torch import nn

# Define the neural network architecture
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(562, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 6) # 6 output classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x

# Initialize the model
model = NeuralNet()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# %% [markdown]
# ## For nuaral net 1

# %%
# Training function
def train_model(model, X_train, y_train, criterion, optimizer, n_epochs=50):
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

        if (epoch+1) % 2 == 0:
            print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

# Train the model
train_model(model, X_train, y_train, criterion, optimizer)

# Evaluate the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    total += y_test.size(0)
    correct += (predicted == y_test).sum().item()

print(f'Accuracy of the model on the test data: {100 * correct / total}%')




# %%
from torch.utils.tensorboard import SummaryWriter
# Initialize the network
net = NeuralNet()
n_inputs = 562
n_classes = 6
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the network to device
net.to(device)

# Define optimizer
optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate)

# Define summary writer
experiment_name = "experiment_har"
writer = SummaryWriter(experiment_name)
writer.add_graph(net, X_train.to(device))
# Initialize iteration number
n_iter = 0

# Define best test accuracy
best_acc = None

# Define loss function
loss_fun = nn.CrossEntropyLoss()

# Lists to store training and test accuracies and losses
train_accuracies = []
test_accuracies = []
train_losses = []
test_losses = []

# Test function
def test(model, test_loader, n_classes, device):
    model.eval()
    correct = 0
    total = 0
    losses = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fun(outputs, labels)
            losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy, sum(losses) / len(losses)

# For each epoch
for cur_epoch in range(50):
    # For each batch
    for inp, gt in train_loader:
        # Move batch to device
        inp = inp.to(device)
        gt = gt.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Get output
        logits = net(inp)

        # Compute loss
        loss = loss_fun(logits, gt)

        # Compute backward
        loss.backward()

        # Update weights
        optimizer.step()

        # Plot
        writer.add_scalar("loss", loss.item(), n_iter)
        n_iter += 1

    # At the end of each epoch, test model on training data
    train_acc, train_loss = test(net, train_loader, n_classes, device)
    train_accuracies.append(train_acc)
    train_losses.append(train_loss)

    # At the end of each epoch, test model on test data
    test_acc, test_loss = test(net, test_loader, n_classes, device)
    test_accuracies.append(test_acc)
    test_losses.append(test_loss)
    writer.add_scalar("train_accuracy", train_acc, cur_epoch)
    writer.add_scalar("test_accuracy", test_acc, cur_epoch)
    writer.add_scalar("train_loss", train_loss, cur_epoch)
    writer.add_scalar("test_loss", test_loss, cur_epoch)

    
    # Check if it is the best model so far
    if best_acc is None or test_acc > best_acc:
        # Define new best accuracy
        best_acc = test_acc

        # Save current model as best
        torch.save({
            'net': net.state_dict(),
            'opt': optimizer.state_dict(),
            'epoch': cur_epoch,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'train_losses': train_losses,
            'test_losses': test_losses
        }, experiment_name + '_best.pth')

    # Save last model
    torch.save({
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': cur_epoch,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'train_losses': train_losses,
        'test_losses': test_losses
    }, experiment_name + '_last.pth')


# %%
import matplotlib.pyplot as plt

# Load the checkpoint
checkpoint = torch.load('experiment_har_last.pth')

# Load accuracies and losses
train_accuracies = checkpoint['train_accuracies']
test_accuracies = checkpoint['test_accuracies']
train_losses = checkpoint['train_losses']
test_losses = checkpoint['test_losses']

# Plot accuracies
plt.figure(figsize=(10, 4))
plt.plot(test_accuracies)
plt.title('Model Test Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# Plot losses
plt.figure(figsize=(10, 4))
plt.plot(train_losses)
plt.plot(test_losses)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# %%
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import seaborn as sns

# Load the best model
model_path = experiment_name + '_best.pth'
checkpoint = torch.load(model_path)
net.load_state_dict(checkpoint['net'])

# Test function
def get_predictions(model, loader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    return all_labels, all_predictions

# Get predictions on the test set
true_labels, predicted_labels = get_predictions(net, test_loader, device)

# Compute metrics
conf_mat = confusion_matrix(true_labels, predicted_labels)
acc = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')
f1 = f1_score(true_labels, predicted_labels, average='weighted')
class_report = classification_report(true_labels, predicted_labels)
# Print metrics
print(f'Accuracy: {acc}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
print(f'Classification report:\n {class_report}')
# Plot confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

