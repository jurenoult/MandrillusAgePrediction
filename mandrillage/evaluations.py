import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from mandrillage.display import display_predictions
from mandrillage.utils import DAYS_IN_YEAR


def compute_std(df, output_dir, display_name="val"):
    predicted_error = {}
    predictions = {}

    # Gather data per id per age range
    for i in tqdm(range(len(df))):
        row = df.iloc[[i]]
        y_true = row["y_true"].values[0]
        y_pred = row["y_pred"].values[0]
        if y_true not in predicted_error:
            predicted_error[y_true] = {}
        id_ = row["id"].values[0]
        if id_ not in predicted_error[y_true]:
            predicted_error[y_true][id_] = []

        abs_error = abs(y_true - y_pred)
        predicted_error[y_true][id_].append(abs_error)

        if y_true not in predictions:
            predictions[y_true] = []
        predictions[y_true].append(y_pred)

    # Compute mean per id per age when multiple photo occurs
    std_by_value = {}
    std_by_value_by_id = {}
    mean_by_value = {}
    # For each unique age value
    for age in predicted_error.keys():
        age_data = predicted_error[age]
        age_pred = predictions[age]

        # Compute std per id with nb photo > 1
        age_stds_by_id = []
        for id_ in age_data.keys():
            if len(age_data[id_]) > 1:
                current_std = np.std(np.array(age_data[id_]))
                age_stds_by_id.append(current_std)
        age_std_by_id = np.mean(age_stds_by_id)
        if not np.isnan(age_std_by_id):
            std_by_value_by_id[age] = age_std_by_id

        # Compute std by age globally
        std_by_value[age] = np.std(np.array(age_pred))
        mean_by_value[age] = np.mean(np.array(age_pred))

    display_predictions(
        predictions,
        os.path.join(output_dir, f"latest_{display_name}_performance"),
    )

    return std_by_value, std_by_value_by_id


def compute_cumulative_scores(df):
    y_pred = np.array(list(df["y_pred"]))
    y_true = np.array(list(df["y_true"]))

    error = abs(y_pred - y_true)
    nb_values = len(y_true)

    cs_values_in_years = [1 / 12, 1 / 6, 1 / 4, 1 / 3, 1 / 2, 1, 2, 3]
    cs_values_in_days = [np.round(val * DAYS_IN_YEAR) for val in cs_values_in_years]

    css = {}
    for i, max_error in enumerate(cs_values_in_days):
        nb_correct = sum(error <= max_error)
        cs = float(nb_correct) / float(nb_values)
        css[f"{i}_CS_{max_error}"] = cs
    return css


def standard_regression_evaluation(y_true, y_pred, name, min_range, max_range):
    steps = [
        DAYS_IN_YEAR / 36,
        DAYS_IN_YEAR / 24,
        DAYS_IN_YEAR / 16,
        DAYS_IN_YEAR / 12,
        DAYS_IN_YEAR / 6,
        DAYS_IN_YEAR / 4,
        DAYS_IN_YEAR / 2,
        DAYS_IN_YEAR,
    ]
    regression_name = f"{name}_regression"
    regression_results = evaluate_regression(
        y_true, y_pred, steps, min_range, max_range, regression_name
    )
    classification_name = f"{name}_as_classification"
    classification_results = evaluate_regression_as_classification(
        y_true, y_pred, steps, min_range, max_range, classification_name
    )

    return format_results(
        regression_name,
        regression_results,
        classification_name,
        classification_results,
        name,
    )


def standard_classification_evaluation(
    y_true, y_prob_pred, class_step, n_classes, name, prob_to_label=np.argmax
):
    y_pred = [prob_to_label(prob) for prob in y_prob_pred]
    classification_name = f"{name}_classification"
    classification_results = evaluate_classification(y_true, y_pred, class_step, n_classes)
    regression_name = f"{name}_as_regression"
    regression_results = evaluate_classification_as_regression(
        y_true,
        y_prob_pred,
        class_step,
        n_classes,
        regression_name,
        prob_to_label=prob_to_label,
    )

    return format_results(
        regression_name,
        regression_results,
        classification_name,
        classification_results,
        name,
    )


def format_results(
    regression_name,
    regression_results,
    classification_name,
    classification_results,
    name,
):
    return {
        name: {
            regression_name: regression_results,
            classification_name: classification_results,
        }
    }


def evaluate_regression(y_true, y_pred, steps, min_range, max_range, name, display=False):
    global_mae, global_std = mae(y_true, y_pred)

    step_results = {}
    for step in steps:
        n_steps = (max_range - min_range) / step
        substep_results = []
        for i in range(int(n_steps)):
            cmin_range = step * i
            cmax_range = step * (i + 1)
            mean_range = (cmin_range + cmax_range) / 2
            mae_value, std = evaluate_by_subrange(y_true, y_pred, cmin_range, cmax_range)

            substep_results.append({"step": mean_range, "mae": mae_value, "std": std})
        step_results[step] = substep_results

    if display:
        plot_regression(global_mae, step_results, min_range, max_range)

    return {f"{name}_mae": global_mae, f"{name}_mae_steps": step_results}


def plot_regression(global_mae, step_results, min_range, max_range):
    # plt.ylim(0, max_range//2)
    plt.plot([min_range, max_range], [global_mae, global_mae], "-", label="MAE")
    for step, step_result in step_results.items():
        x, y = list(zip(*step_result))
        plt.plot(x, y, "-", label=f"step_{int(step)}")
    plt.legend()
    plt.show()


def evaluate_by_subrange(y_true, y_pred, min_range, max_range):
    selection = np.logical_and(y_true >= min_range, y_true <= max_range)
    y_true_subrange = y_true[selection]
    y_pred_subrange = y_pred[selection]
    if y_true_subrange.shape[0] == 0 or y_pred_subrange.shape[0] == 0:
        return 0.0, 0.0
    return mae(y_true_subrange, y_pred_subrange)


def mae(y_true, y_pred):
    diff = abs(y_true - y_pred)
    return np.mean(diff), np.std(diff)


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # Normalize the confusion matrix

    plt.figure(figsize=(16, 10))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()


def evaluate_classification(y_true, y_pred, class_step, n_classes, display=False):
    labels = range_to_labels(class_step, n_classes)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    if display:
        plot_confusion_matrix(y_true, y_pred, labels)

    results = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "f1_per_class": f1_per_class,
        "labels": labels,
    }
    return results


def evaluate_classification_as_regression(
    y_true_classes, y_pred_prob_classes, class_step, n_classes, name, prob_to_label
):
    labels = range_to_labels(class_step, n_classes)
    y_true_scalars = classification_to_regression(y_true_classes, labels)
    y_pred_scalars = classification_to_regression(prob_to_label(y_pred_prob_classes), labels)
    # y_pred_weigthed_scalars = classification_prob_to_regression(
    #     y_pred_prob_classes, labels
    # )
    min_range = 0
    max_range = class_step * n_classes
    return {
        "argmax": evaluate_regression(
            y_true_scalars, y_pred_scalars, [class_step, 90], min_range, max_range, name
        ),
        # "weighted_scalars": evaluate_regression(
        #     y_true_scalars,
        #     y_pred_weigthed_scalars,
        #     [class_step, 90],
        #     min_range,
        #     max_range,
        #     name,
        # ),
    }


def evaluate_regression_as_classification(
    y_true_scalars, y_pred_scalars, class_steps, min_range, max_range, name
):
    all_results = {}
    for class_step in class_steps:
        n_classes = int(np.ceil(max_range / class_step))
        current_name = f"{name}_{int(class_step)}"
        y_true_classes = regression_to_classification(
            y_true_scalars, class_step, min_range, max_range
        )
        y_pred_classes = regression_to_classification(
            y_pred_scalars, class_step, min_range, max_range
        )

        results = evaluate_classification(y_true_classes, y_pred_classes, class_step, n_classes)

        all_results[current_name] = results
    return all_results


def regression_to_classification(y_scalars, class_step, min_range, max_range):
    # Clip values between min an max to fit in classes
    y_scalars = [min(max_range, max(scalar, min_range)) for scalar in y_scalars]
    y_classes = [scalar_to_class(scalar, class_step) for scalar in y_scalars]
    return y_classes


def scalar_to_class(scalar, class_step):
    class_index = scalar / class_step
    class_index = max(0, np.ceil(class_index) - 1)
    return class_index


def classification_prob_to_regression(y_classes_prob, labels):
    ranges = labels_to_range(labels)
    y_scalars = [prob_class_to_scalar(y_class_prob, ranges) for y_class_prob in y_classes_prob]
    return np.array(y_scalars)


def prob_class_to_scalar(class_prob, ranges):
    np_mid_ranges = np.array([(stop + start) / 2 for start, stop in ranges])
    total_prob = np.sum(class_prob)
    # assert abs(total_prob - 1.0) < 1e-4
    # Compute weighted probability
    scalar = np.sum(np_mid_ranges * np.array(class_prob))
    return scalar


def classification_to_regression(y_classes, labels):
    ranges = labels_to_range(labels)
    y_scalars = [class_to_scalar(class_value, ranges) for class_value in y_classes]
    return np.array(y_scalars)


def class_to_scalar(class_index, ranges):
    start, stop = ranges[class_index]
    return (stop + start) / 2


def range_to_labels(class_step, n_classes):
    labels = [f"Age [{int(class_step*i)};{int(class_step*(i+1))}]" for i in range(n_classes)]
    return labels


def labels_to_range(labels):
    ranges = []
    for label in labels:
        label = label.split("[")[1]
        label = label.split("]")[0]
        start, stop = label.split(";")
        ranges.append((int(start), int(stop)))
    return ranges
