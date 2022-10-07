import pandas as pd
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
import joblib

from src.NestedKFoldAlgoTesterOriginal import NestedKFoldAlgoTester

if __name__ == '__main__':
    data = pd.read_csv('smart_grid_stability_augmented.csv')
    data['stabf'] = data['stabf'].apply(lambda x: x == 'stable')

    X = data.iloc[:, :12]
    y = data.stabf

    std_metrics = ['accuracy', 'f1', 'f1_neg']
    algo_tester = NestedKFoldAlgoTester(X, y, inner_splits=10, outer_splits=10, metrics=std_metrics)

    param_grid_tree = [
        {'max_depth': [50, 100, 1000],
         'min_samples_leaf': [1, 4, 64]}
    ]

    param_grid_forest = [
        {'max_depth': [25, 50, 100],
         'n_estimators': [5, 25, 100, 200],
         'min_samples_leaf': [1, 4, 64]}
    ]

    param_grid_knn = [
        {'n_neighbors': [1, 6, 9, 18]}
    ]

    param_grid_lgb = [
        {'max_depth': [25, 50, 100],
         'n_estimators': [400, 1600, 3200],
         'min_child_weight': [4, 16, 64]}
    ]

    # original:
    param_grid_ada = [
        {'n_estimators': [25, 50, 100, 500],
         'learning_rate': [.1, 1.0, 2.0]}
    ]

    algo_tester.algo_test(DecisionTreeClassifier(), param_grid=param_grid_tree, registry_name='DecisionTree')
    algo_tester.algo_test(RandomForestClassifier(n_jobs=4), param_grid=param_grid_forest, registry_name='RandomForest')
    algo_tester.algo_test(KNeighborsClassifier(n_jobs=4), param_grid=param_grid_knn, registry_name='KNN')
    algo_tester.algo_test(LGBMClassifier(n_jobs=4), param_grid=param_grid_lgb, registry_name='LightGBM')
    algo_tester.algo_test(AdaBoostClassifier(), param_grid=param_grid_ada, registry_name='AdaBoost')

    registry = algo_tester.results
    joblib.dump(registry, "results/resutlado_consolidado_adaboost_FINAL"+str(int(datetime.now().timestamp()))+".pkl")




