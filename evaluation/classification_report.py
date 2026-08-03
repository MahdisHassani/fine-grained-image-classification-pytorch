import os
import pandas as pd
from sklearn.metrics import classification_report

def save_classification_report(
    y_true,
    y_pred,
    save_path="outputs/classification_report.csv",
    class_names=None
):

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    df = pd.DataFrame(report).transpose()

    df.to_csv(save_path)

    return df