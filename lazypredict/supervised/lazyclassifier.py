import pandas as pd
from tqdm import tqdm
import time
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)
from lazypredict import CLASSIFIERS
from lazypredict.supervised.preprocessing import preprocess_data


class LazyClassifier:
    # TODO: add loved clasifiers by default
    # TODO: add the chance to use kfolds instead of train_test_split
    # TODO: add a Metrics class to declutter this even further
    # TODO: make it compatible with Grid Search
    # TODO: do something about random state variable
    # TODO: add variable to ensure you filter results by desired variable
    # TODO: implement predictions
    # TODO: Improve provide_models function
    """
    This module helps in fitting to all the classification algorithms that are available in Scikit-learn
    Parameters
    ----------
    verbose : Bool, optional (default=False)
    classifiers : list, optional (default="all")
        When function is provided, trains the chosen classifier(s).

    Examples
    --------
    >>> from lazypredict.supervised.lazyclassifier import LazyClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_breast_cancer()
    >>> X = data.data
    >>> y = data.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)
    >>> clf = LazyClassifier()
    >>> models = clf.fit(X_train, X_test, y_train, y_test)
    >>> model_dictionary = clf.provide_models(X_train,X_test,y_train,y_test)
    >>> models
    | Model                          |   Accuracy |   Balanced Accuracy |   ROC AUC |   F1 Score |   Time Taken |
    |:-------------------------------|-----------:|--------------------:|----------:|-----------:|-------------:|
    | LinearSVC                      |   0.989474 |            0.987544 |  0.987544 |   0.989462 |    0.0150008 |
    | SGDClassifier                  |   0.989474 |            0.987544 |  0.987544 |   0.989462 |    0.0109992 |
    | MLPClassifier                  |   0.985965 |            0.986904 |  0.986904 |   0.985994 |    0.426     |
    | Perceptron                     |   0.985965 |            0.984797 |  0.984797 |   0.985965 |    0.0120046 |
    | LogisticRegression             |   0.985965 |            0.98269  |  0.98269  |   0.985934 |    0.0200036 |
    | LogisticRegressionCV           |   0.985965 |            0.98269  |  0.98269  |   0.985934 |    0.262997  |
    | SVC                            |   0.982456 |            0.979942 |  0.979942 |   0.982437 |    0.0140011 |
    | CalibratedClassifierCV         |   0.982456 |            0.975728 |  0.975728 |   0.982357 |    0.0350015 |
    | PassiveAggressiveClassifier    |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0130005 |
    | LabelPropagation               |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0429988 |
    | LabelSpreading                 |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0310006 |
    | RandomForestClassifier         |   0.97193  |            0.969594 |  0.969594 |   0.97193  |    0.033     |
    | GradientBoostingClassifier     |   0.97193  |            0.967486 |  0.967486 |   0.971869 |    0.166998  |
    | QuadraticDiscriminantAnalysis  |   0.964912 |            0.966206 |  0.966206 |   0.965052 |    0.0119994 |
    | HistGradientBoostingClassifier |   0.968421 |            0.964739 |  0.964739 |   0.968387 |    0.682003  |
    | RidgeClassifierCV              |   0.97193  |            0.963272 |  0.963272 |   0.971736 |    0.0130029 |
    | RidgeClassifier                |   0.968421 |            0.960525 |  0.960525 |   0.968242 |    0.0119977 |
    | AdaBoostClassifier             |   0.961404 |            0.959245 |  0.959245 |   0.961444 |    0.204998  |
    | ExtraTreesClassifier           |   0.961404 |            0.957138 |  0.957138 |   0.961362 |    0.0270066 |
    | KNeighborsClassifier           |   0.961404 |            0.95503  |  0.95503  |   0.961276 |    0.0560005 |
    | BaggingClassifier              |   0.947368 |            0.954577 |  0.954577 |   0.947882 |    0.0559971 |
    | BernoulliNB                    |   0.950877 |            0.951003 |  0.951003 |   0.951072 |    0.0169988 |
    | LinearDiscriminantAnalysis     |   0.961404 |            0.950816 |  0.950816 |   0.961089 |    0.0199995 |
    | GaussianNB                     |   0.954386 |            0.949536 |  0.949536 |   0.954337 |    0.0139935 |
    | NuSVC                          |   0.954386 |            0.943215 |  0.943215 |   0.954014 |    0.019989  |
    | DecisionTreeClassifier         |   0.936842 |            0.933693 |  0.933693 |   0.936971 |    0.0170023 |
    | NearestCentroid                |   0.947368 |            0.933506 |  0.933506 |   0.946801 |    0.0160074 |
    | ExtraTreeClassifier            |   0.922807 |            0.912168 |  0.912168 |   0.922462 |    0.0109999 |
    | CheckingClassifier             |   0.361404 |            0.5      |  0.5      |   0.191879 |    0.0170043 |
    | DummyClassifier                |   0.512281 |            0.489598 |  0.489598 |   0.518924 |    0.0119965 |
    """

    def __init__(
        self,
        verbose=False,
        predictions=False,
        random_state=42,
        classifiers="all",
    ):
        self.verbose = verbose
        self.predictions = predictions
        self.models = {}
        self.random_state = random_state
        self.classifiers = classifiers

    def fit(self, X_train, X_test, y_train, y_test):
        """Fit Classification algorithms to X_train and y_train, predict and score on X_test, y_test.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        scores : Pandas DataFrame
            Returns test set metrics of all the models in a Pandas DataFrame.
        """
        accuracies = []
        balanced_accuracies = []
        roc_aucs = []
        f1s = []
        precisions = []
        recalls = []
        model_names = []
        times = []
        predictions = {}

        preprocessing_pipeline = preprocess_data(X_train, X_test)

        if self.classifiers == "all":
            self.classifiers = CLASSIFIERS

        else:
            try:
                temp_list = []
                for classifier in self.classifiers:
                    full_name = (classifier.__name__, classifier)
                    temp_list.append(full_name)
                self.classifiers = temp_list

            except Exception as exception:
                print(exception)
                print("Invalid Classifier(s)")

        for name, model in tqdm(self.classifiers):
            start = time.time()
            try:
                if "random_state" in model().get_params().keys():
                    pipe = Pipeline(
                        steps=[
                            ("preprocessing_pipeline", preprocessing_pipeline),
                            ("classifier", model(random_state=self.random_state)),
                        ]
                    )
                else:
                    pipe = Pipeline(
                        steps=[
                            ("preprocessing_pipeline", preprocessing_pipeline),
                            ("classifier", model()),
                        ]
                    )

                pipe.fit(X_train, y_train)
                self.models[name] = pipe

                y_pred = pipe.predict(X_test)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred, normalize=True)
                b_accuracy = balanced_accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")

                try:
                    roc_auc = roc_auc_score(y_test, y_pred)
                except Exception as exception:
                    roc_auc = None
                    print("ROC AUC couldn't be calculated for " + name)
                    print(exception)

                model_names.append(name)
                accuracies.append(accuracy)
                balanced_accuracies.append(b_accuracy)
                precisions.append(precision)
                recalls.append(recall)
                roc_aucs.append(roc_auc)
                f1s.append(f1)
                times.append(time.time() - start)

                if self.verbose:
                    print(
                        {
                            "Model": name,
                            "Accuracies": accuracy,
                            "Balanced accuracies": b_accuracy,
                            "ROC AUC": roc_auc,
                            "Precision": precisions,
                            "Recall": recalls,
                            "f1s Score": f1,
                            "Time taken": time.time() - start,
                        }
                    )
                if self.predictions:
                    predictions[name] = y_pred

            except Exception as exception:
                print(name + " model failed to execute")
                print(exception)

        scores = pd.DataFrame(
            {
                "Model": model_names,
                "Accuracies": accuracies,
                "Balanced accuracies": balanced_accuracies,
                "ROC AUC": roc_aucs,
                "Precision": precisions,
                "Recall": recalls,
                "F1 Score": f1s,
                "Time Taken": times,
            }
        )

        scores = scores.sort_values(by="F1 Score", ascending=False).set_index("Model")

        if self.predictions:
            # TODO: not implemented yet
            predictions_df = pd.DataFrame.from_dict(predictions)

        return scores

    def provide_models(self, X_train, X_test, y_train, y_test):
        # TODO: Make this work only if you already train the models?
        """
        This function returns all the model objects trained in fit function.
        If fit is not called already, then we call fit and then return the models.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        models: dict-object,
            Returns a dictionary with each model pipeline as value
            with key as name of models.
        """
        if len(self.models.keys()) == 0:
            self.fit(X_train, X_test, y_train, y_test)

        return self.models


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=123
    )

    clf = LazyClassifier()

    models = clf.fit(X_train, X_test, y_train, y_test)
    model_dictionary = clf.provide_models(X_train, X_test, y_train, y_test)
    print(models, 3 * "\n")
