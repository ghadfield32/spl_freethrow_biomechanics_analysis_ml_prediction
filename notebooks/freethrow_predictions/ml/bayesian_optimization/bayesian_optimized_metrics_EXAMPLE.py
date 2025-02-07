
from skopt import gp_minimize
from skopt.space import Real
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

def prepare_data():
    """Prepare training data."""
    np.random.seed(42)
    X_test = pd.DataFrame({
        'knee_max_angle': np.random.uniform(40, 140, 200),
        'wrist_max_angle': np.random.uniform(0, 90, 200),
        'elbow_max_angle': np.random.uniform(30, 160, 200),
    })
    y_test = pd.Series(np.random.choice([0, 1], size=200))

    features = ['knee_max_angle', 'wrist_max_angle', 'elbow_max_angle']
    X_train = X_test[features]
    y_train = y_test

    # Debug: Check the dataset details
    print(f"X_train shape: {X_train.shape}")
    print(f"X_train sample:\n{X_train.head()}")
    print(f"y_train sample:\n{y_train.head()}")
    return X_train, y_train, features


def train_model(X_train, y_train):
    """Train a Decision Tree Classifier."""
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Debug: Check feature importances
    print(f"Feature importances: {clf.feature_importances_}")
    return clf


def define_search_space(features):
    """Define the search space for optimization."""
    spaces = {
        'knee_max_angle': Real(40, 140, name='knee_angle'),
        'wrist_max_angle': Real(30, 90, name='wrist_angle'),
        'elbow_max_angle': Real(30, 160, name='elbow_angle')
    }
    return [spaces[feature] for feature in features]


def objective_function(clf):
    """Create an objective function for Bayesian Optimization."""
    def objective(params):
        knee, wrist, elbow = params
        input_df = pd.DataFrame([[knee, wrist, elbow]], 
                                columns=['knee_max_angle', 'wrist_max_angle', 'elbow_max_angle'])
        success_prob = clf.predict_proba(input_df)[0, 1]
        # Debug: Log evaluation details
        print(f"Evaluating: knee={knee:.2f}, wrist={wrist:.2f}, elbow={elbow:.2f}, success_prob={success_prob:.2f}")
        return -success_prob  # Negative for minimization
    return objective


def perform_optimization(objective, space):
    """Perform Bayesian Optimization."""
    res = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=50,
        n_random_starts=10,
        random_state=42
    )
    # Debug: Log optimization result details
    print(f"Optimization result: {res}")
    return res


def calculate_baselines(X_train):
    """Calculate baseline values for each feature."""
    baselines = {col: X_train[col].mean() for col in X_train.columns}
    print(f"Baseline values: {baselines}")
    return baselines


def compare_results(features, baselines, results):
    """Compare optimal values with baselines."""
    print("\nOptimization Results:")
    for feature, baseline, optimal in zip(features, baselines.values(), results.x):
        difference = optimal - baseline
        print(f"{feature} - Optimal: {optimal:.2f}, Baseline: {baseline:.2f}, Difference: {difference:.2f}")


# Main workflow
X_train, y_train, features = prepare_data()
clf = train_model(X_train, y_train)
space = define_search_space(features)
objective = objective_function(clf)
res = perform_optimization(objective, space)
baselines = calculate_baselines(X_train)
compare_results(features, baselines, res)
