from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from text_entailment__util.project_log import logger_msg



def logistic_regression(
    X_train,
    X_test,
    y_train,
    y_test,
    max_iter = 1000
):

    try:
        logger_msg("--- Logistic Regression Results ---")
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(X_train, y_train)

        # Predict on the test set
        y_pred = classifier.predict(X_test)

        # Evaluate the model
        logger_msg(classification_report(y_test, y_pred))

    except Exception as e:

        logger_msg("Baseline Models | Logistic Regression | Error in the code.")

    return None


def svm(
    X_train,
    X_test,
    y_train,
    y_test,
    kernel='linear'
):

    try:
        logger_msg("--- SVM Results ---")
        classifier = SVC(kernel=kernel)
        classifier.fit(X_train, y_train)

        # Predict on the test set
        y_pred = classifier.predict(X_test)

        # Evaluate the model
        logger_msg(classification_report(y_test, y_pred))

    except Exception as e:

        logger_msg("Baseline Models | SVM | Error in the code.")
    
    return None


def random_forest(
    X_train,
    X_test,
    y_train,
    y_test,
    n_estimators=100,
    random_state=42
):

    try:
        logger_msg("--- Random Forest Results ---")
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        classifier.fit(X_train, y_train)

        # Predict on the test set
        y_pred = classifier.predict(X_test)

        # Evaluate the model
        logger_msg(classification_report(y_test, y_pred))

    except Exception as e:

        logger_msg("Baseline Models | Random Forest | Error in the code.")

    return None