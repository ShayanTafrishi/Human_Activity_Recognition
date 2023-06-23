import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib import rcParams
from torchsummary import summary
import sys
from io import StringIO


# Set Page Title and Icon
st.set_page_config(page_title="Activity Recognition Dashboard", page_icon="📊")

# Sidebar Controls
st.sidebar.title("Controls")
show_data = st.sidebar.checkbox('Show Raw Data', False)
show_training_pie_chart = st.sidebar.checkbox('Show Training Data Analysis', False)
show_performance = st.sidebar.checkbox('Show Model Performance', False)
show_network_structure = st.sidebar.checkbox('Network structure', False)
# Select model
selected_model = st.sidebar.selectbox("Select Model", ["NeuralNet1", "NeuralNet2"])

# Load the data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

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

# Main Header
st.title('Activity Recognition Dashboard 📊')

# About the Dataset
st.markdown("""
## About the Dataset 📝
This dataset is collected from experiments that consist of monitoring human activities through smartphone sensors. It has features like accelerometer and gyroscope readings in different directions. The activities that are classified in this dataset are walking, walking upstairs, walking downstairs, sitting, standing, and laying.
""")

# Show data
if show_data:
    st.subheader('Raw Data 📋')
    st.dataframe(df_train)

# Show training data set figures
if show_training_pie_chart:
    st.subheader("Training Data Analysis 📊")
    activities = df_train['Activity'].value_counts()

    plt.figure(figsize=(10,10))
    plt.pie(activities, labels=activities.index, autopct='%1.1f%%', startangle=90)
    plt.title('Balance of different activity types')

    st.pyplot(plt)

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

    st.pyplot(plt)

if show_performance or show_network_structure:
    import torch
    # Import the neural network classes
    from NeuralNet1 import NeuralNet
    from NeuralNet2 import NeuralNet2
    # Load the model
    if selected_model == "NeuralNet1":
        model = NeuralNet()
        checkpoint = torch.load('experiment_har_best.pth')
    elif selected_model == "NeuralNet2":
        model = NeuralNet2()
        checkpoint = torch.load('experiment_har2_best.pth')

    model.load_state_dict(checkpoint['net'])
    model.eval()

    # Make predictions on test data
    X_test = torch.tensor(X_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.long)
    output = model(X_test)
    _, predicted = torch.max(output.data, 1)

    # Show model's performance
    if show_performance:
        st.subheader('Model Performance 📈')
        conf_mat = confusion_matrix(y_test, predicted)
        acc = accuracy_score(y_test, predicted)
        precision = precision_score(y_test, predicted, average='weighted')
        recall = recall_score(y_test, predicted, average='weighted')
        f1 = f1_score(y_test, predicted, average='weighted')
        st.markdown(f"""
        - **Accuracy**: {acc:.3f}
        - **Precision**: {precision:.3f}
        - **Recall**: {recall:.3f}
        - **F1 Score**: {f1:.3f}
        """)

        # Plot confusion matrix
        fig, ax = plt.subplots()
        sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        st.pyplot(fig)
        st.markdown(""" ## Best performing model 🏆 """)
    
        
        
        
        # Load accuracies and losses
        train_accuracies = checkpoint['train_accuracies']
        test_accuracies = checkpoint['test_accuracies']
        train_losses = checkpoint['train_losses']
        test_losses = checkpoint['test_losses']

       
        numEpochs = checkpoint['epoch']
        best_accuracy = test_accuracies[numEpochs]
        best_loss = test_losses[numEpochs]

        st.write(f'Best model epoch: {numEpochs}')
        st.write(f'Best model accuracy: {best_accuracy}%')
        st.write(f'Best model loss: {best_loss}')
        
    if show_network_structure:
        st.subheader('Network Structure 🧠')
        if selected_model == "NeuralNet1":
            model = NeuralNet()
        
        elif selected_model == "NeuralNet2":
            model = NeuralNet2()
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        # Assuming `model` is your PyTorch model and it's already on the correct device
        summary(model, input_size=(562, ))

        # Reset stdout
        sys.stdout = old_stdout

        # Use captured output in Streamlit
        st.text(mystdout.getvalue())
