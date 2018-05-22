import numpy as np
from pandas import DataFrame

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from Preprocessing import Preprocessing

# chi_selected_features = ['AveKnow', 'AveCarelessness', 'mean_hint', 'mean_timeGreater10SecAndNextActionRight',
#                          'mean_bottomHint', 'mean_correct', 'mean_past8BottomOut', 'mean_manywrong', 'mean_hintCount',
#                          'mean_hintTotal']

# chi_selected_features = ['AveKnow', 'AveCarelessness', 'mean_hint', 'mean_timeGreater10SecAndNextActionRight',
#                          'mean_bottomHint', 'mean_correct']

chi_selected_features = ['AveKnow']


def get_model(cv=None):
    # prepare the necessary model classes and search spaces
    estimator = Pipeline([
        ('scale', RobustScaler()),
        ('model', LinearSVC(max_iter=10000, class_weight="balanced"))
    ])

    # search spaces for different model classes
    gbrt = {
        'model': [GradientBoostingClassifier()],
        'model__n_estimators': [2 ** i for i in range(1, 11)],
        'model__learning_rate': [2 ** i for i in range(-10, 10)],
    }

    svc = {
        'model': [SVC()],
        'model__C': [10 ** i for i in np.linspace(-6, 6, 20)],
        'model__gamma': [10 ** i for i in np.linspace(-6, 6, 20)],
    }

    # linsvc = {
    #     'model': [LinearSVC(max_iter=10000, class_weight="balanced")],
    #     'model__C': [10 ** i for i in np.linspace(-6, 6, 20)],
    # }

    linsvc = {
        # 'model': [LinearSVC(max_iter=10000, class_weight="balanced")],
        'model__C': [10 ** i for i in np.linspace(-6, 6, 20)],
    }

    knn = {
        'model': [KNeighborsClassifier()],
        'model__n_neighbors': range(1, 44),
    }
    dectree = {
        'model': [DecisionTreeClassifier()],
        'model__max_depth': range(1, 20),
        'model__min_samples_split': [2 ** i for i in range(-20, -1)],
    }

    # this class does search over all parameter spaces for parameter
    # combination which yields the best validation loss
    model = GridSearchCV(
        estimator=estimator,
        param_grid=[linsvc, svc, dectree, knn],
        n_jobs=-1,
        scoring='roc_auc',
        verbose=1,
        cv=cv
    )

    # a class implementing trivial model - guess either at random
    # or a class that is most likely
    dummy_model = GridSearchCV(
        estimator=estimator,
        param_grid=[{
            'model': [DummyClassifier()],
            'model__strategy': ['most_frequent', 'uniform'],
        }],
        scoring='roc_auc',
        n_jobs=-1,
        cv=cv
    )

    # TODO add a return parameter for dummy model

    return model


def test_model_actionwise(X, y, x_competition):
    # assumes X_train/X_test has stud id

    # preparing train/test dataset

    train_ix, test_ix = next(GroupShuffleSplit(n_splits=2, train_size=.75).split(X, y, groups=X.index.get_level_values(0).values))
    X_train, X_test, y_train, y_test = X.iloc[train_ix], X.iloc[test_ix], y.take(train_ix), y.take(test_ix)

    # pre.raw_dataset.to_csv("Debug Dataset/raw_dataset.csv", index=False)
    # pre.test_dataset.to_csv("Debug Dataset/test_dataset.csv", index=False)
    # pre.label_dataset.to_csv("Debug Dataset/label_dataset.csv", index=False)
    # pre.per_action_dataset.to_csv("Debug Dataset/per_action_dataset.csv", index=False)
    # pre.per_action_dataset_summ.to_csv("Debug Dataset/per_action_dataset_summ.csv", index=False)
    # pre.per_stud_dataset.to_csv("Debug Dataset/per_stud_dataset.csv", index=False)
    #
    # X.to_csv("Debug Dataset/X.csv", index=False)
    # DataFrame(y).to_csv("Debug Dataset/y.csv", index=False)
    # X_train.to_csv("Debug Dataset/X_train.csv", index=False)
    # X_test.to_csv("Debug Dataset/X_test.csv", index=False)
    # DataFrame(y_train).to_csv("Debug Dataset/y_train.csv", index=False)
    # DataFrame(y_test).to_csv("Debug Dataset/y_test.csv", index=False)

    gkf = list(GroupKFold(n_splits=5).split(X_train, y_train, groups=X_train.index.get_level_values(0).values))
    model = get_model(gkf)

    model.fit(X_train.values, y_train)

    per_action_pred = model.predict(X_test.values)

    per_action_validation_score = model.score(X_test.values, y_test)
    print("Per-Action Validation Score: %f" % per_action_validation_score)

    DataFrame(per_action_pred).to_csv("Debug Dataset/per_action_pred.csv", index=False)

    per_stud_pred = action2stud_pred(X_test.index.get_level_values(0).values, per_action_pred)

    per_stud_labels = DataFrame({'ITEST_id': X_test.index.get_level_values(0).values, 'isSTEM': y_test}).groupby("ITEST_id")[
        'isSTEM'].first().values

    # DataFrame(per_stud_labels).to_csv("Debug Dataset/per_stud_labels.csv", index=False)

    roc_test_score = roc_auc_score(per_stud_labels, per_stud_pred['prediction'])
    mse_test_score = mean_squared_error(per_stud_labels, per_stud_pred['prediction'])

    print("Test score (ROC): %s" % roc_test_score)
    print("Test score (MSE): %s" % mse_test_score)
    print("Model score: %s" % model.best_score_)
    print("Best parameters: %s" % model.best_params_)
    print("Best model: %s" % model.best_estimator_)
    print("Competition Test Score: %s" % (1 - mse_test_score + roc_test_score))
    print("CV results %s" % model.cv_results_)

    # print_scores(model, per_stud_labels, per_stud_pred['prediction'])

    # per_stud_pred['isSTEM'] = per_stud_labels

    # per_stud_pred.to_csv("predicted_labels.csv", index=False)
    #
    # per_stud_pred.to_csv("Debug Dataset/per_stud_pred.csv", index=False)

    # fitting best model on all data and predicting on competition dataset

    # comp_fit = model.best_estimator_.fit(X.drop('ITEST_id', axis=1).values, y)
    # comp_fit = model.best_estimator_.fit(X.values, y)
    #
    # # comp_pred = comp_fit.predict(x_competition.drop('ITEST_id', axis=1).values)
    # comp_pred = comp_fit.predict(x_competition.values)
    #
    # comp_per_stud_pred = action2stud_pred(x_competition['ITEST_id'], comp_pred)
    #
    # comp_per_stud_pred.to_csv("predicted_labels_comp.csv", index=False)


    return per_action_validation_score, roc_test_score, mse_test_score, model.best_score_

def action2stud_pred(stud_ids, pred_action_preds):
    # assumes that the pred_action_labels contains predicted labels for stud_ids,
    # i.e., pred_action_labels[0] is predicted label for one of the actions of stud_ids[0]

    # converting per action predictions to per student predictions
    # based on summing all per action predictions for each student divided by total number of actions for that student

    per_action_label = DataFrame({'ITEST_id': stud_ids, 'prediction': pred_action_preds})
    per_stud_label = per_action_label.groupby(level="ITEST_id").size()

    per_stud_label = DataFrame(per_stud_label, columns=["n_actions"])
    per_stud_label['sum_prediction'] = per_action_label.groupby(level="ITEST_id")['prediction'].sum()
    per_stud_label['ITEST_id'] = per_stud_label.index.values
    per_stud_label['prediction'] = per_stud_label['sum_prediction'] / per_stud_label['n_actions']

    return per_stud_label


pre = Preprocessing()

# X_competition, _ = pre.load_data(return_val_tst_set=True, time_gap=300)

# X_competition = X_competition[chi_selected_features]

X, y = pre.load_data(time_gap=300)

X = X[chi_selected_features]

test_model_actionwise(X, y.values, None)
