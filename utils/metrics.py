from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def evaluate_predictions(y_true, y_pred, phase, class_names):
    
    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }
    
    if phase == "test":
        precision_recall_fscore_support = classification_report(
            y_true,
            y_pred
        )
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        
        return metrics, cm, precision_recall_fscore_support
    
    return metrics