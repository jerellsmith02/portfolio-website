Python 3.12.6 (tags/v3.12.6:a4a2d2b, Sep  6 2024, 20:11:23) [MSC v.1940 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
import os
# Define the path to the dataset
... data_path = r'D:\python\MLRD-Machine-Learning-Ransomware-Detection-master\data_file.csv'
... # Check if the dataset file exists
... if not os.path.isfile(data_path):
...     print(f"{data_path} not found. Please check the file path.")
... else:
...     # Load the dataset
...     df = pd.read_csv(data_path)
...     # Print the columns to verify the correct ones
...     print("Columns in dataset:", df.columns.tolist())
...     # Preprocess the dataset by dropping unnecessary columns
...     # Update these column names as per your actual dataset structure
...     if 'FileName' in df.columns and 'md5Hash' in df.columns and 'Benign' in df.columns:
...         X = df.drop(['FileName', 'md5Hash', 'Benign'], axis=1).values
...         y = df['Benign'].values  # Assuming 'Benign' is the target column where 1 = benign and 0 = ransomware
...     else:
...         print("Error: Expected columns not found in the dataset.")
...         raise KeyError("Check the column names and update the script accordingly.")
...     # Scale the features using StandardScaler
...     scaler = StandardScaler()
...     X_scaled = scaler.fit_transform(X)
...     # Split data into training and testing sets (80% training, 20% testing)
...     X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
...     # Initialize lists to store accuracy and loss results
...     models = [
...         ('Linear Regression', LinearRegression(), False),  # Regression Model (not for classification)
...         ('Logistic Regression', LogisticRegression(max_iter=2000), True),  # Increased max_iter
...         ('K-Nearest Neighbors (KNN)', KNeighborsClassifier(), True)
...     ]
...     train_accuracies = []
...     test_accuracies = []
...     train_losses = []
...     test_losses = []
...     # Function to train and evaluate the model
    def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, is_classification=True):
        # Train the model
        model.fit(X_train, y_train)
        # Predict on training and testing data
        if is_classification:
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            # Calculate accuracy
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            # Calculate loss using log_loss for classification tasks
            y_train_prob = model.predict_proba(X_train)
            y_test_prob = model.predict_proba(X_test)
            train_loss = log_loss(y_train, y_train_prob)
            test_loss = log_loss(y_test, y_test_prob)
        else:  # Linear regression doesn't have a classification loss/accuracy, we use R^2 as proxy
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_accuracy = model.score(X_train, y_train)
            test_accuracy = model.score(X_test, y_test)
            # Since log_loss is inappropriate for regression, we'll use mean squared error as loss proxy
            train_loss = np.mean((y_train_pred - y_train) ** 2)
            test_loss = np.mean((y_test_pred - y_test) ** 2)
        # Print accuracy and loss results
        print(f"\n{model_name} - Training Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
        print(f"{model_name} - Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        return train_accuracy, test_accuracy, train_loss, test_loss
    # Loop through each model and evaluate its performance
    for model_name, model, is_classification in models:
        train_acc, test_acc, train_loss, test_loss = train_and_evaluate_model(model, X_train, y_train, X_test, y_test,
                                                                              model_name, is_classification)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    # Part 2: Plot Accuracy and Loss
    # Plotting accuracy and loss in line plot format
    plt.figure(figsize=(14, 6))
    # Labels for the models
    x_labels = [name for name, _, _ in models]
    # Plotting Training and Test Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(x_labels, train_accuracies, label='Training Accuracy', marker='o', linestyle='-', color='blue')
    plt.plot(x_labels, test_accuracies, label='Validation Accuracy', marker='o', linestyle='-', color='orange')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Models')
    plt.ylabel('Accuracy / R^2 for Linear Regression')
    plt.legend()
    # Plotting Training and Test Loss
    plt.subplot(1, 2, 2)
    plt.plot(x_labels, train_losses, label='Training Loss', marker='o', linestyle='-', color='blue')
    plt.plot(x_labels, test_losses, label='Validation Loss', marker='o', linestyle='-', color='orange')
    plt.title('Training and Validation Loss')
    plt.xlabel('Models')
    plt.ylabel('Loss (Log Loss / MSE for Linear Regression)')
    plt.legend()
    plt.tight_layout()
    plt.show()
