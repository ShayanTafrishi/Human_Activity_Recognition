# Activity Recognition Project

This project aims to perform activity recognition using two different neural network architectures: NeuralNet1 and NeuralNet2. The dataset used for training and testing consists of smartphone sensor readings for various activities.

## Prerequisites

Before running the code, make sure you have the following installed:

- Python 3.x
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Streamlit

## Usage

1. Run the following code cell to train and test NeuralNet:

    ```
    # Cell for NeuralNet

    # ... (code for data loading, preprocessing, and model definition)

    train_model(model, X_train, y_train, criterion, optimizer, n_epochs=100)
    test_model(model, X_test, y_test)
    ```

    This will train the NeuralNet1 model using the provided dataset and display the test results.

2. Once the training and testing for NeuralNet1 is complete, run the following code cell to train and test NeuralNet2:

    ```
    # Cell for NeuralNet2

    # ... (code for data loading, preprocessing, and model definition)

    train_model(model, X_train, y_train, criterion, optimizer, n_epochs=100)
    test_model(model, X_test, y_test)
    ```

    This will train the NeuralNet2 model using the same dataset and display the test results.

# Cleaning the Paths
If you want to start fresh and remove all the previously saved models and logs, you can delete the following directories and files:

path_to_logs: The directory containing the TensorBoard logs.
* experiment_har_last.pth: The saved model for Neural Net 1.
* experiment_har2_last.pth: The saved model for Neural Net 2.
* Any other saved models or checkpoints you may have created.
Please exercise caution when deleting files and directories to avoid unintended data loss.

## Results

The results of each model can be viewed in the console output after running the corresponding code cells. Additionally, Streamlit is used to provide a graphical user interface for visualizing the data, training progress, and model performance. Launch the Streamlit app by running the following command:

! streamlit run streamlitApp.py 

This can be done in the last cell of the notebook or in the terminal of the EXAM folder.

## Stramlit

This will open a web browser with the Streamlit app, where you can interact with the various features of the project.


# Contributors

s.tafrishi@campus.unimib.it

