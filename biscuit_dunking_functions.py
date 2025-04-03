'''
Artificial Intelligence and Deep Learning
Programming Project 2 - Using Machine Learning to Investigate Biscuit Dunking Data

Written by Elliot Ayliffe
Student ID : 2046374

Module containing the functions required to run the corresponding Jupyter Notebook.

Functions for:
- Preprocessing
- Training Machine Learning Models (classification and regression)
- Evaluating ML Models 
- Plotting results
- Washburn Equation 
'''

# Import libraries 
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance



#-----------------------------------------------------------------------------------------------------------------------------------------
# PREPROCESSING & MACHINE LEARNING MODEL FUNCTIONS
#-----------------------------------------------------------------------------------------------------------------------------------------

## Function to preprocess a dataset before being used to train a ML model
def preprocessing(X, y, test_size=0.25, stratify=False, scaling=True):
    """
    Preprocesses the dataset:
    - Splits the data into training and testing sets (with optional stratification for classification tasks)
    - Scales the features (standardization, optional)

    Args:
        X (pandas.DataFrame): Feature set (input variables)
        y (pandas.Series): Target variable (labels for classification or continuous values for regression)
        test_size (float): Proportion of the data set to use for testing (default is 0.25)
        stratify (bool): Whether to apply stratification to the train/test split. Ensures even distribution of labels in training and testing sets 
                        (should be applied for classification tasks, default is False)
        scaling (bool): Whether to apply standardization to the feature set X. Uses Standard Scaler (default is True)

    Returns:
        numpy.ndarrays: X_train, X_test, y_train, y_test. The dataset preprocessed split into training and testing sets ready for ML training.
    """
    # Determine whether to stratify the target variable or not 
    strat_parameter = y if stratify else None

    # Split into train-test datasets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=strat_parameter)

    # Scale the features (if True)
    if scaling:
        scale = StandardScaler()
        X_train = scale.fit_transform(X_train)
        X_test = scale.transform(X_test)

    return X_train, X_test, y_train, y_test



## Functions to train machine learning algorithms (classification and regression)
def train_random_forest_model(X_train, y_train, task, optimise_model=False):
    """ 
    Trains a Random Forest Model (either Classification or Regression) with optional model optimisation (hyperparameter tuning)
    using GridSearchCV. The task type, classification or regression, is specified by the user.

    Args:
        X_train (numpy.ndarray): Training feature set 
        y_train (numpy.ndarray): Training target variables
        task (str): The type of task you are doing ('classification' or 'regression). This specifies the type of Random Forest Model that will be trained
        optimise_model (bool): Whether to apply hyperparameter tuning using GridsearchCV 

    Returns: 
        trained_rf_model: Trained random forest model (either classification or regression)
    """
    # Define model based on the user input (task)
    if task == 'classification':
        model = RandomForestClassifier(random_state=42)
        scoring='accuracy'      # Set the scoring metric

    elif task == 'regression':
        model = RandomForestRegressor(random_state=42)
        scoring='neg_mean_squared_error'            # Using the negative as Grid search tries to maximise the score
   
    # Perform hyperparameter tuning (these apply to both classification and regression tasks)
    if optimise_model:
        
        # Define the parameter grid to test 
        parameter_grid = {
            'n_estimators': [100, 200, 300],        # Number of trees in the forest
            'max_depth': [10, 20, None],            # Maximum depth of the tree
            'min_samples_split': [2, 5, 10],        # Minimum number of samples required to split an internal node
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Use a cross-validated grid search to tune the hyperparameters 
        grid_search = GridSearchCV(estimator=model, param_grid=parameter_grid, cv=5, n_jobs=-1, scoring=scoring, verbose=1)
        # Train model
        grid_search.fit(X_train, y_train)

        # Print best parameters and cross-validation score 
        print(f"Optimal parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.2f}")

        trained_rf_model = grid_search.best_estimator_

    else:           # For no hyperparameter tuning (without optimization)
        trained_rf_model = model.fit(X_train, y_train)

    return trained_rf_model 



def train_SVM_model(X_train, y_train, task, optimise_model=False):
    """ 
    Trains a Support Vector Model (SVC - classification or SVR - regression) with optional optimisation (hyperparameter tuning)
    using GridSearchCV. The task type, classification or regression, is specified by the user.

    Args:
        X_train (numpy.ndarray): Training feature set 
        y_train (numpy.ndarray): Training target variables
        task (str): The type of task you are doing ('classification' or 'regression). This specifies the type of SVM Model that will be trained.
        optimise_model (bool): Whether to apply hyperparameter tuning using GridsearchCV     

    Returns:
        trained_svm_model: Trained SVM model (either classification or regression)
    """
    # Define model based on the user input (task)
    if task == 'classification':
        model = SVC(random_state=42)
        scoring='accuracy'      # Set the scoring metric

    elif task == 'regression':
        model = SVR()
        scoring='neg_mean_squared_error'            # Using the negative as Grid search tries to maximise the score

    if optimise_model:
        
        # Define the parameter grid to test 
        parameter_grid = {
            'C': [0.1, 1, 10, 100],                             # Regularisation parameter (lower=wider margin decision boundary)
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],     # Specifies the kernel type for transforming the data   
            'gamma': ['scale', 'auto'],                         # Kernel coefficient (controls the influence of single points)
            'degree': [3, 4, 5],                                # Degree of polynomial kernel (higher= more complex decision boundary)
            'coef0': [0.0, 0.5, 1.0]                            # Independent term in kernel function (used in poly and sigmoid, controls the influence of the bias term)
        }
        
        # Use a cross-validated grid search to tune the hyperparameters 
        grid_search = GridSearchCV(estimator=model, param_grid=parameter_grid, cv=5, n_jobs=-1, scoring=scoring, verbose=1)
        # Train model
        grid_search.fit(X_train, y_train)

        # Print best parameters and cross-validation score 
        print(f"Optimal parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.2f}")

        trained_svm_model = grid_search.best_estimator_

    else:           # For no hyperparameter tuning (without optimization)
        trained_svm_model = model.fit(X_train, y_train)

    return trained_svm_model



def train_MLP_classifier(X_train, y_train, optimise_model=False):
    """ 
    Trains a Multi-layer perceptron (MLP) classification model with optional optimisation (parameter/hyperparameter tuning)
    MLP is a type of simple artificial neural network.

    Args:
        X_train (numpy.ndarray): Training feature set 
        y_train (numpy.ndarray): Training target variables
        optimise_model (bool): Whether to apply parameter/hyperparameter tuning using GridsearchCV     

    Returns:
        trained_mlp_classifier: Trained MLP classification model
    """
    # Define model 
    mlp_classifier = MLPClassifier(random_state=42, max_iter=2000)

    if optimise_model:
        
        # Define the parameter grid to test 
        parameter_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],       # Various hidden layer architectures   
            'alpha': [0.0001, 0.001],                              # Strength of regularisation (L2)
            'learning_rate': ['constant', 'invscaling'],           # how the learning rate evolves
            'activation': ['relu', 'tanh']
        }

        # Use a cross-validated grid search to tune the hyperparameters 
        grid_search = GridSearchCV(estimator=mlp_classifier, param_grid=parameter_grid, cv=5, n_jobs=-1, scoring='accuracy',verbose=1)
        # Train model
        grid_search.fit(X_train, y_train)

        # Print best parameters and cross-validation score 
        print(f"Optimal parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.2f}")

        trained_mlp_classifier = grid_search.best_estimator_

    else:           # For no hyperparameter tuning (without optimization)
        trained_mlp_classifier = mlp_classifier.fit(X_train, y_train)

    return trained_mlp_classifier




#-----------------------------------------------------------------------------------------------------------------------------------------
# MODEL EVALUATION FUNCTIONS
#-----------------------------------------------------------------------------------------------------------------------------------------

# Function to make predictions on train/test data, compute and display the evaluation metrics for classification models
def classifier_evaluation(trained_model, X_train, X_test, y_train, y_test, algorithm_name):
    """ 
    Evaluation function for classification models.
    Makes predictions on the training and testing datasets, then computes and displays
    the evaluation metrics (accuracy, precision, recall, f1-score).

    Args: 
        trained_model: The trained sklearn classifier
        X_train (numpy.ndarray): Training feature set 
        X_test (numpy.ndarray): Testing feature set
        y_train (numpy.ndarray): Training target variables
        y_test (numpy.ndarray): Testing target variables 
        algorithm_name (str): Name of the classification algorithm

    Returns:
        Displays the training and testing accuracy along with the classification report.
        accuracy_testing (float): Accuracy achieved on the testing data
    """
    # Make predictions on training and testing data 
    y_pred_training = trained_model.predict(X_train)
    y_pred_testing = trained_model.predict(X_test)

    # Computing accuracies 
    accuracy_training = accuracy_score(y_train, y_pred_training)
    accuracy_testing = accuracy_score(y_test, y_pred_testing)

    # print evaluation metrics 
    print(f"\nEvaluation Metrics for {algorithm_name}:\n")
    print(f"Accuracy on the training data: {accuracy_training*100:.1f}% ")
    print(f"Accuracy on the testing data: {accuracy_testing*100:.1f}% ")
    print(classification_report(y_test, y_pred_testing))             # precision, recall, F1-score

    return accuracy_testing


# Function to make predictions on train/test data, compute and display the evaluation metrics for regression models
def regressor_evaluation(trained_model, X_train, X_test, y_train, y_test, algorithm_name):
    """ 
    Evaluation function for regression models.
    Makes predictions on the training and testing datasets, then computes and displays
    the evaluation metrics (RMSE, R²).

    Args: 
        trained_model: The trained sklearn classifier
        X_train (numpy.ndarray): Training feature set 
        X_test (numpy.ndarray): Testing feature set
        y_train (numpy.ndarray): Training target variables
        y_test (numpy.ndarray): Testing target variables 
        algorithm_name (str): Name of the regression algorithm

    Returns:
        Displays the training and testing evaluation metrics.
        rmse_testing (float): RMSE (root mean squared error) achieved on the testing data
        r_squared_testing (float): R² score achieved on the testing data 
        y_pred_testing (numpy.ndarray): The predicted values for the testing data 
    """
    # Make predictions on training and testing data 
    y_pred_training = trained_model.predict(X_train)
    y_pred_testing = trained_model.predict(X_test)

    # Compute RMSE and R² scores
    rmse_training = np.sqrt(mean_squared_error(y_train, y_pred_training))
    r_squared_training = r2_score(y_train, y_pred_training)
    rmse_testing = np.sqrt(mean_squared_error(y_test, y_pred_testing))
    r_squared_testing = r2_score(y_test, y_pred_testing)

    # Display evaluation metrics 
    print(f"\nEvaluation Metrics for {algorithm_name}:\n")
    print(f"RMSE on the training data: {rmse_training:.2e}")
    print(f"R² on the training data: {r_squared_training:.4f}\n")
    print(f"RMSE on the testing data: {rmse_testing:.2e}")
    print(f"R² on the testing data: {r_squared_testing:.4f}")

    return rmse_testing, r_squared_testing, y_pred_testing




#-----------------------------------------------------------------------------------------------------------------------------------------
# RESULTS PLOTTING FUNCTIONS 
#-----------------------------------------------------------------------------------------------------------------------------------------


# Function to plot the accuracies of the 3 classification models (with and without optimisation)
def heatmap_classifier_accuracies(test_accuracies):
    """ 
    Plots a heatmap of the test accuracies for the 3 different classification models, 
    comparing initial vs optimised to visualise any improvements, and highlighting the 
    best classification model for this dataset.

    Args: 
        test_accuracies (dict): Dictionary of the 3 classification models with the accuracy results (normal and optimised)
                                e.g. { 'Model Name': [test_accuracy_initial], [test_accuracy_optimised]}

    Returns:
        Heatmap displaying the test accuracies of the 3 classification models
    """
    # Convert dictionary to DataFrame 
    df = pd.DataFrame(test_accuracies, index=['Non-optimised', 'Optimised']).T * 100

    # plot heatmap 
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df, annot=True, fmt=".1f", cmap='RdYlGn', vmin=80, vmax=100, cbar_kws={'label': 'Accuracy (%)'}, ax=ax)

    # Format plot
    ax.set_title("Test Accuracy: Classification Models", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.show()




  # Function to plot permutation importance (method for assessing the feature importance)
def plot_permutation_importance(X, X_test, y_test, trained_rf_classifier, trained_svc, trained_mlp_classifier):
    """ 
    Computes and plots a bar graph of the permutation importance for each of the classification models.
    Permutation importance is a method to assess the feature importance by measuring the reduction in model
    performance when a feature is randomly shuffled. It can be applied to all model which is why it is used 
    over impurity-based feature importance.

    Args:
        X (pandas.DataFrame): The original dataframe to retrieve feature names 
        X_test (numpy.ndarray): Testing feature set
        y_test (numpy.ndarray): Testing target variables 
        trained_rf_classifier,
        trained_svc,
        trained_mlp_classifier: Trained classification models
    """
    # Feature names
    feature_names = X.columns

    # Compute the permutation importance for each classifier 
    rf_perm = permutation_importance(trained_rf_classifier, X_test, y_test, n_repeats=20, random_state=42)
    svc_perm = permutation_importance(trained_svc, X_test, y_test, n_repeats=20, random_state=42)
    mlp_perm = permutation_importance(trained_mlp_classifier, X_test, y_test, n_repeats=20, random_state=42)

    # plot bar graph
    x = np.arange(len(feature_names))
    w = 0.25

    fig, ax = plt.subplots()
    ax.bar(x - w, rf_perm.importances_mean, w, label='Random Forest', color='blue')
    ax.bar(x, svc_perm.importances_mean, w, label='SVC', color='green')
    ax.bar(x + w, mlp_perm.importances_mean, w, label='MLP', color='red')

    # Format plot 
    ax.set_title("Permutation Importance for Classifiers")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names)
    ax.set_xlabel("Features")
    ax.set_ylabel("Permutation Importance")
    ax.legend()
    plt.show()  



## Functions to visualise the difference in pore radius across biscuit type 
def boxplots(df, x_name, y_name, ax):
    """ 
    Plots a boxplot from a given DataFrame and specified column names.
    This is for comparing the pore radius between biscuit type. 

    Args:
        df (pandas.DataFrame): The dataset containing the columns you wish to plot
        x_name (str): the name of the column for the categories (e.g. 'biscuit')
        y_name (str): the name of the column for the numerical values (e.g. 'r' - pore radius)
        ax: the specific axes for the plot (formatting)

    Returns:
        Displays the boxplot of the users inputs.
    """
    # Using seaborn for a nicer plot 
    sns.boxplot(x=x_name, y=y_name, data=df, palette="Set1", hue=x_name, ax=ax)
    ax.set_xlabel('Biscuit Type')
    ax.set_ylabel('Pore Radius (m)')
    


def histogram(df, x_name, y_name, ax):
    """ 
    Plots a histogram from a given DataFrame and specified columns.
    This is for comparing the distribution of the pore radius for each biscuit type.

    Args:
        df (pandas.DataFrame): The dataset containing the columns you wish to plot
        x_name (str): the name of the column for the categories (e.g. 'biscuit')
        y_name (str): the name of the column for the numerical values (e.g. 'r' - pore radius)
        ax: the specific axes for the plot (formatting)

    Returns:
        Displays the histogram from the users inputs  
    """
    # Using seaborn for a nicer plot 
    # Set KDE (kernel density estimation) to True. This helps visualise the underlying distribution by plotting a smooth continous curve.
    sns.histplot(data=df, x=y_name, hue=x_name, palette="Set1", bins=50, kde=True, ax=ax)
    ax.set_xlabel('Pore Radius (m)')
    ax.set_ylabel('Frequency') 




# Functions to visualise the comparison between Random Forest Regression and Washburn Equation
def plot_wb_rf_comparison(yl_test, yr_test, rf_pred_L, wb_pred_L, rf_pred_r, wb_pred_r, metrics):
    """ 
    This function plots two scatter plots side-by-side comparing Random forest regression and Washburn equation predictions.
    1. Actual 'L' values vs predicted 'L' values 
    2. Actual 'r' values vs predicted 'r' values 

    This figure visually compares the performance of both methods when predicting continuous variables.

    Args:
        yl_test (numpy.ndarray): The Actual measured values for 'L' from the test dataset  
        yr_test (numpy.ndarray): The Actual measured values for 'r' from the test dataset 
        rf_pred_L (numpy.ndarray): Predicted values of 'L' by the Random Forest regressor 
        wb_pred_L (numpy.ndarray): Predicted values of 'L' using the Washburn equation 
        rf_pred_r (numpy.ndarray): Predicted values of 'r' by the Random Forest regressor 
        wb_pred_r (numpy.ndarray): Predicted values of 'r' using the Washburn equation 
        metrics (list): List containing the evaluation metrics (RMSE and R-squared scores)
    """
    # unpack the metrics list 
    rf_reg_rmse, rf_reg_r2, rmse_washburn_L, r2_washburn_L, rf_reg_rmse_r, rf_reg_r2_r, rmse_washburn_r, r2_washburn_r = metrics

    fig, ax = plt.subplots(1,2, figsize=(14,6))

    # Plot actual L vs predicted L 
    ax[0].scatter(yl_test, rf_pred_L, label='Random Forest Prediction', color='blue', s=25, marker='o')       # RF predictions 
    ax[0].scatter(yl_test, wb_pred_L, label='Washburn Prediction', color='red', s=25, marker='s')             # Washburn predictions
    ax[0].plot([yl_test.min(), yl_test.max()], [yl_test.min(), yl_test.max()], label='Perfect Prediction', linestyle='-', color='black') # Perfect prediction line for reference (100% accurate)
    ax[0].set_xlabel("Actual L (m)")
    ax[0].set_ylabel("Predicted L (m)")
    ax[0].set_title(f"Actual vs Predicted Absorption Length 'L'")
    ax[0].legend()

    # Add Metrics to the figure as text 
    ax[0].text(0.95, 0.15, f"RF: RMSE = {rf_reg_rmse:.2e},   R² = {rf_reg_r2:.4f}", color='blue', transform=ax[0].transAxes, ha='right', va='bottom')
    ax[0].text(0.95, 0.05, f"Washburn: RMSE = {rmse_washburn_L:.2e},   R² = {r2_washburn_L:.4f}", color='red', transform=ax[0].transAxes, ha='right', va='bottom')

    # Plot actual r vs predicted r 
    ax[1].scatter(yr_test, rf_pred_r, label='Random Forest Prediction', color='blue', s=25, marker='o')       # RF predictions 
    ax[1].scatter(yr_test, wb_pred_r, label='Washburn Prediction', color='red', s=25, marker='s')             # Washburn predictions
    ax[1].plot([yr_test.min(), yr_test.max()], [yr_test.min(), yr_test.max()], label='Perfect Prediction', linestyle='-', color='black') # Perfect prediction line for reference (100% accurate)
    ax[1].set_xlabel("Actual r (m)")
    ax[1].set_ylabel("Predicted r (m)")
    ax[1].set_title(f"Actual vs Predicted Pore Radius 'r'")

    # Add Metrics to the figure as text 
    ax[1].text(0.95, 0.15, f"RF: RMSE = {rf_reg_rmse_r:.2e},   R² = {rf_reg_r2_r:.4f}", color='blue', transform=ax[1].transAxes, ha='right', va='bottom')
    ax[1].text(0.95, 0.05, f"Washburn: RMSE = {rmse_washburn_r:.2e},   R² = {r2_washburn_r:.4f}", color='red', transform=ax[1].transAxes, ha='right', va='bottom')


#-----------------------------------------------------------------------------------------------------------------------------------------
# WASHBURN EQUATION FUNCTIONS
#-----------------------------------------------------------------------------------------------------------------------------------------

# Functions to compute the Washburn Equation
def washburn_L(df):
    """ 
    Computes the absorption length 'L' using the Washburn equation 
    for each sample in a given data frame.

    Args:
        df (pandas.DataFrame): data frame containing the parameters needed for the Washburn equation 

    Returns: 
        L (pandas.Series): The calculated L values
    """
    # Extract the feature columns from the dataframe
    gamma = df['gamma']
    phi = df['phi']
    eta = df['eta']
    t = df['t']
    r = df['r']

    # Compute L using the Washburn equation 
    L = np.sqrt((gamma * r * t * np.cos(phi)) / (2 * eta))

    return L

def washburn_r(df):
    """ 
    Computes the pore radius 'r' using the Washburn equation 
    for each sample in a given dataframe.

    Args:
        df (pandas.DataFrame): data frame containing the parameters needed for the Washburn equation 

    Returns:
        r (pandas.Series): The calculated r values 
    """
    # Extract the feature columns from the dataframe
    gamma = df['gamma']
    phi = df['phi']
    eta = df['eta']
    t = df['t']
    L = df['L']

    # Compute r using the Washburn equation
    r = (2 * eta * L**2) / (gamma * t * np.cos(phi))

    return r
