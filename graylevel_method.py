import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from PIL import Image

# Specify the path to the saved pickle files
val_input_file = "graylevels_validation_b.pkl"
test_input_file = "graylevels_testing.pkl"
val_input_file_normed = "graylevels_normed_validation_b.pkl"
test_input_file_normed = "graylevels_normed_testing.pkl"

# Load the data from the pickle files
with open(val_input_file, 'rb') as f:
    development_metadata = pickle.load(f)
with open(test_input_file, 'rb') as f:
    test_metadata = pickle.load(f)
with open(val_input_file_normed, 'rb') as f:
    development_normed_metadata = pickle.load(f)
with open(test_input_file_normed, 'rb') as f:
    test_normed_metadata = pickle.load(f)

def calculate_graylevel_proportions(data: np.ndarray, data_length: int, denorm: bool = True) -> np.ndarray:
    """
    Calculates the proportions of gray levels in the data.

    Args:
        data (np.ndarray): Input data.
        data_length (int): The length of the data.
        denorm (bool, optional): Whether to denormalize the data by multiplying by 255. Default is True.

    Returns:
        np.ndarray: A 2D array with the proportions of gray levels for each data point.
    """
    if denorm:
        denormed_data = data * 255.0
    else:
        denormed_data = data

    bin_edges = np.arange(0, 256 + 1, 16)
    proportions = np.zeros(shape=(data_length, 16))

    for i, datapoint in enumerate(denormed_data):
        counts, _ = np.histogram(datapoint, bins=bin_edges)
        proportions[i] = counts

    return proportions

if __name__ == '__main__':
    test_metadata['gray_level_props'] = calculate_graylevel_proportions(test_metadata['gray_levels'], 5)
    test_normed_metadata['gray_level_props'] = calculate_graylevel_proportions(test_normed_metadata['gray_levels'], 5, denorm=False)

    test_metadata['gray_level_props'] = test_metadata['gray_level_props'] / 50176
    test_normed_metadata['gray_level_props'] = test_normed_metadata['gray_level_props'] / 50176


    development_metadata['gray_level_props'] = calculate_graylevel_proportions(development_metadata['gray_levels'], 5)
    development_normed_metadata['gray_level_props'] = calculate_graylevel_proportions(development_normed_metadata['gray_levels'], 5, denorm=False)

    development_metadata['gray_level_props'] = development_metadata['gray_level_props'] / 50176
    development_normed_metadata['gray_level_props'] = development_normed_metadata['gray_level_props'] / 50176

    ############################## LOGISTIC REGRESSION GRIDSEARCH ##############################
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['saga']
    }

    logreg = LogisticRegression(random_state=1120, max_iter=500)
    grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, scoring='roc_auc_ovr', cv=5, n_jobs=36)
    grid_search_normed = GridSearchCV(estimator=logreg, param_grid=param_grid, scoring='roc_auc_ovr', cv=5, n_jobs=36)
    
    print('BEGINNING LOGISTIC REGRESSION SEARCH')
    grid_search.fit(development_metadata['gray_level_props'], (development_metadata['density']-1).ravel())
    grid_search_normed.fit(development_normed_metadata['gray_level_props'], (development_normed_metadata['density']-1).ravel())

    print('Raw graylevels:  ', grid_search.best_score_)
    print( grid_search.best_params_)

    print('Normed graylevels:  ', grid_search_normed.best_score_)
    print( grid_search_normed.best_params_)

    development_metadata['logreg_preds'] = grid_search.predict_proba(development_metadata['gray_level_props'])
    development_normed_metadata['logreg_preds'] = grid_search_normed.predict_proba(development_normed_metadata['gray_level_props'])
    test_metadata['logreg_preds'] = grid_search.predict_proba(test_metadata['gray_level_props'])
    test_normed_metadata['logreg_preds'] = grid_search_normed.predict_proba(test_normed_metadata['gray_level_props'])

    ############################## MLP GRIDSEARCH ##############################
    mlp = MLPClassifier(random_state=1120, max_iter=500)
    param_grid = {
        'hidden_layer_sizes': [(128,), (256,), (512,)],
        'activation': ['relu', 'tanh'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
    }

    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, scoring='roc_auc_ovr', cv=5, n_jobs=36)
    grid_search_normed = GridSearchCV(estimator=mlp, param_grid=param_grid, scoring='roc_auc_ovr', cv=5, n_jobs=36)

    print('BEGINNING MLP SEARCH')
    grid_search.fit(development_metadata['gray_level_props'], (development_metadata['density']-1).ravel())
    grid_search_normed.fit(development_normed_metadata['gray_level_props'], (development_normed_metadata['density']-1).ravel())

    print('Raw graylevels:  ', grid_search.best_score_)
    print( grid_search.best_params_)
    print('Normed graylevels:  ', grid_search_normed.best_score_)
    print( grid_search_normed.best_params_)

    development_metadata['mlp_preds'] = grid_search.predict_proba(development_metadata['gray_level_props'])
    development_normed_metadata['mlp_preds'] = grid_search_normed.predict_proba(development_normed_metadata['gray_level_props'])
    test_metadata['mlp_preds'] = grid_search.predict_proba(test_metadata['gray_level_props'])
    test_normed_metadata['mlp_preds'] = grid_search_normed.predict_proba(test_normed_metadata['gray_level_props'])

    ############################## RANDOM FOREST GRIDSEARCH ##############################
    rf_classifier = RandomForestClassifier(random_state=1120)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, scoring='roc_auc_ovr', cv=5, n_jobs=36)
    grid_search_normed = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, scoring='roc_auc_ovr', cv=5, n_jobs=36)

    print('BEGINNING RANDOM FOREST SEARCH')
    grid_search.fit(development_metadata['gray_level_props'], (development_metadata['density']-1).ravel())
    grid_search_normed.fit(development_normed_metadata['gray_level_props'], (development_normed_metadata['density']-1).ravel())

    print('Raw graylevels:  ', grid_search.best_score_)
    print( grid_search.best_params_)
    print('Normed graylevels:  ', grid_search_normed.best_score_)
    print( grid_search_normed.best_params_)

    development_metadata['rf_preds'] = grid_search.predict_proba(development_metadata['gray_level_props'])
    development_normed_metadata['rf_preds'] = grid_search_normed.predict_proba(development_normed_metadata['gray_level_props'])
    test_metadata['rf_preds'] = grid_search.predict_proba(test_metadata['gray_level_props'])
    test_normed_metadata['rf_preds'] = grid_search_normed.predict_proba(test_normed_metadata['gray_level_props'])

    with open('graylevels_predictions_validation_b.pkl', 'wb') as file:
        pickle.dump(development_metadata, file)

    with open('graylevels_normed_predictions_validation_b.pkl', 'wb') as file:
        pickle.dump(development_normed_metadata, file)

    with open('graylevels_predictions_testing.pkl', 'wb') as file:
        pickle.dump(test_metadata, file)

    with open('graylevels_normed_predictions_testing.pkl', 'wb') as file:
        pickle.dump(test_normed_metadata, file)




