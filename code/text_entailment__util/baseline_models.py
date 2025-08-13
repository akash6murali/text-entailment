from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from text_entailment__util.project_log import logger_msg


from sklearn.metrics import confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{model_name}_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()

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
        auc = roc_auc_score(y_test, y_pred)
        logger_msg(f"ROC AUC Score (OvR): {auc:.4f}")
        logger_msg(classification_report(y_test, y_pred))

        plot_confusion_matrix(y_test, y_pred, "Logistic Regression")

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
        auc = roc_auc_score(y_test, y_pred)
        logger_msg(f"ROC AUC Score (OvR): {auc:.4f}")
        logger_msg(classification_report(y_test, y_pred))

        plot_confusion_matrix(y_test, y_pred, "Logistic Regression")

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
        auc = roc_auc_score(y_test, y_pred)
        logger_msg(f"ROC AUC Score (OvR): {auc:.4f}")
        
        logger_msg(classification_report(y_test, y_pred))

        plot_confusion_matrix(y_test, y_pred, "Logistic Regression")

    except Exception as e:

        logger_msg("Baseline Models | Random Forest | Error in the code.")

    return None