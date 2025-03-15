import marimo

__generated_with = "0.11.20"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Chapter 3 – Classification**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        _This notebook contains all the sample code and solutions to the exercises in chapter 3._
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <table align="left">
          <td>
            <a href="https://colab.research.google.com/github/ageron/handson-ml3/blob/main/03_classification.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
          </td>
          <td>
            <a target="_blank" href="https://kaggle.com/kernels/welcome?src=https://github.com/ageron/handson-ml3/blob/main/03_classification.ipynb"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>
          </td>
        </table>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Setup
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This project requires Python 3.7 or above:
        """
    )
    return


@app.cell
def _():
    import sys

    assert sys.version_info >= (3, 7)
    return (sys,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It also requires Scikit-Learn ≥ 1.0.1:
        """
    )
    return


@app.cell
def _():
    from packaging import version
    import sklearn

    assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
    return sklearn, version


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Just like in the previous chapter, let's define the default font sizes to make the figures prettier:
        """
    )
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And let's create the `images/classification` folder (if it doesn't already exist), and define the `save_fig()` function which is used through this notebook to save the figures in high-res for the book:
        """
    )
    return


@app.cell
def _(plt):
    from pathlib import Path

    IMAGES_PATH = Path() / "images" / "classification"
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)

    def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
        path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)
    return IMAGES_PATH, Path, save_fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # MNIST
        """
    )
    return


@app.cell
def _():
    from sklearn.datasets import fetch_openml

    mnist = fetch_openml('mnist_784', as_frame=False)
    return fetch_openml, mnist


@app.cell
def _(mnist):
    # extra code – it's a bit too long
    print(mnist.DESCR)
    return


@app.cell
def _(mnist):
    mnist.keys()  # extra code – we only use data and target in this notebook
    return


@app.cell
def _(mnist):
    X, y = mnist.data, mnist.target
    X
    return X, y


@app.cell
def _(X):
    X.shape
    return


@app.cell
def _(y):
    y
    return


@app.cell
def _(y):
    y.shape
    return


@app.cell
def _():
    28 * 28
    return


@app.cell
def _(X, plt, save_fig):
    def plot_digit(image_data):
        image = image_data.reshape(28, 28)
        plt.imshow(image, cmap="binary")
        plt.axis("off")

    some_digit = X[0]
    plot_digit(some_digit)
    save_fig("some_digit_plot")  # extra code
    plt.show()
    return plot_digit, some_digit


@app.cell
def _(y):
    y[0]
    return


@app.cell
def _(X, plot_digit, plt, save_fig):
    # extra code – this cell generates and saves Figure 3–2
    plt.figure(figsize=(9, 9))
    for idx, image_data in enumerate(X[:100]):
        plt.subplot(10, 10, idx + 1)
        plot_digit(image_data)
    plt.subplots_adjust(wspace=0, hspace=0)
    save_fig("more_digits_plot", tight_layout=False)
    plt.show()
    return idx, image_data


@app.cell
def _(X, y):
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Training a Binary Classifier
        """
    )
    return


@app.cell
def _(y_test, y_train):
    y_train_5 = (y_train == '5')  # True for all 5s, False for all other digits
    y_test_5 = (y_test == '5')
    return y_test_5, y_train_5


@app.cell
def _(X_train, y_train_5):
    from sklearn.linear_model import SGDClassifier

    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train_5)
    return SGDClassifier, sgd_clf


@app.cell
def _(sgd_clf, some_digit):
    sgd_clf.predict([some_digit])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Performance Measures
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Measuring Accuracy Using Cross-Validation
        """
    )
    return


@app.cell
def _(X_train, sgd_clf, y_train_5):
    from sklearn.model_selection import cross_val_score

    cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
    return (cross_val_score,)


@app.cell
def _(X_train, sgd_clf, y_train_5):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.base import clone

    skfolds = StratifiedKFold(n_splits=3)  # add shuffle=True if the dataset is not
                                           # already shuffled
    for train_index, test_index in skfolds.split(X_train, y_train_5):
        clone_clf = clone(sgd_clf)
        X_train_folds = X_train[train_index]
        y_train_folds = y_train_5[train_index]
        X_test_fold = X_train[test_index]
        y_test_fold = y_train_5[test_index]

        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        n_correct = sum(y_pred == y_test_fold)
        print(n_correct / len(y_pred))
    return (
        StratifiedKFold,
        X_test_fold,
        X_train_folds,
        clone,
        clone_clf,
        n_correct,
        skfolds,
        test_index,
        train_index,
        y_pred,
        y_test_fold,
        y_train_folds,
    )


@app.cell
def _(X_train, y_train_5):
    from sklearn.dummy import DummyClassifier

    dummy_clf = DummyClassifier()
    dummy_clf.fit(X_train, y_train_5)
    print(any(dummy_clf.predict(X_train)))
    return DummyClassifier, dummy_clf


@app.cell
def _(X_train, cross_val_score, dummy_clf, y_train_5):
    cross_val_score(dummy_clf, X_train, y_train_5, cv=3, scoring="accuracy")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Confusion Matrix
        """
    )
    return


@app.cell
def _(X_train, sgd_clf, y_train_5):
    from sklearn.model_selection import cross_val_predict

    y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    return cross_val_predict, y_train_pred


@app.cell
def _(y_train_5, y_train_pred):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_train_5, y_train_pred)
    cm
    return cm, confusion_matrix


@app.cell
def _(confusion_matrix, y_train_5):
    y_train_perfect_predictions = y_train_5  # pretend we reached perfection
    confusion_matrix(y_train_5, y_train_perfect_predictions)
    return (y_train_perfect_predictions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Precision and Recall
        """
    )
    return


@app.cell
def _(y_train_5, y_train_pred):
    from sklearn.metrics import precision_score, recall_score

    precision_score(y_train_5, y_train_pred)  # == 3530 / (687 + 3530)
    return precision_score, recall_score


@app.cell
def _(cm):
    # extra code – this cell also computes the precision: TP / (FP + TP)
    cm[1, 1] / (cm[0, 1] + cm[1, 1])
    return


@app.cell
def _(recall_score, y_train_5, y_train_pred):
    recall_score(y_train_5, y_train_pred)  # == 3530 / (1891 + 3530)
    return


@app.cell
def _(cm):
    # extra code – this cell also computes the recall: TP / (FN + TP)
    cm[1, 1] / (cm[1, 0] + cm[1, 1])
    return


@app.cell
def _(y_train_5, y_train_pred):
    from sklearn.metrics import f1_score

    f1_score(y_train_5, y_train_pred)
    return (f1_score,)


@app.cell
def _(cm):
    # extra code – this cell also computes the f1 score
    cm[1, 1] / (cm[1, 1] + (cm[1, 0] + cm[0, 1]) / 2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Precision/Recall Trade-off
        """
    )
    return


@app.cell
def _(sgd_clf, some_digit):
    y_scores = sgd_clf.decision_function([some_digit])
    y_scores
    return (y_scores,)


@app.cell
def _(y_scores):
    threshold = 0
    y_some_digit_pred = (y_scores > threshold)
    return threshold, y_some_digit_pred


@app.cell
def _(y_some_digit_pred):
    y_some_digit_pred
    return


@app.cell
def _(y_scores):
    # extra code – just shows that y_scores > 0 produces the same result as
    #              calling predict()
    y_scores > 0
    return


@app.cell
def _(y_scores):
    threshold_1 = 3000
    y_some_digit_pred_1 = y_scores > threshold_1
    y_some_digit_pred_1
    return threshold_1, y_some_digit_pred_1


@app.cell
def _(X_train, cross_val_predict, sgd_clf, y_train_5):
    y_scores_1 = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')
    return (y_scores_1,)


@app.cell
def _(y_scores_1, y_train_5):
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores_1)
    return precision_recall_curve, precisions, recalls, thresholds


@app.cell
def _(plt, precisions, recalls, save_fig, threshold_1, thresholds):
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall', linewidth=2)
    plt.vlines(threshold_1, 0, 1.0, 'k', 'dotted', label='threshold')
    idx_1 = (thresholds >= threshold_1).argmax()
    plt.plot(thresholds[idx_1], precisions[idx_1], 'bo')
    plt.plot(thresholds[idx_1], recalls[idx_1], 'go')
    plt.axis([-50000, 50000, 0, 1])
    plt.grid()
    plt.xlabel('Threshold')
    plt.legend(loc='center right')
    save_fig('precision_recall_vs_threshold_plot')
    plt.show()
    return (idx_1,)


@app.cell
def _(idx_1, plt, precisions, recalls, save_fig):
    import matplotlib.patches as patches
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, linewidth=2, label='Precision/Recall curve')
    plt.plot([recalls[idx_1], recalls[idx_1]], [0.0, precisions[idx_1]], 'k:')
    plt.plot([0.0, recalls[idx_1]], [precisions[idx_1], precisions[idx_1]], 'k:')
    plt.plot([recalls[idx_1]], [precisions[idx_1]], 'ko', label='Point at threshold 3,000')
    plt.gca().add_patch(patches.FancyArrowPatch((0.79, 0.6), (0.61, 0.78), connectionstyle='arc3,rad=.2', arrowstyle='Simple, tail_width=1.5, head_width=8, head_length=10', color='#444444'))
    plt.text(0.56, 0.62, 'Higher\nthreshold', color='#333333')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.axis([0, 1, 0, 1])
    plt.grid()
    plt.legend(loc='lower left')
    save_fig('precision_vs_recall_plot')
    plt.show()
    return (patches,)


@app.cell
def _(precisions, thresholds):
    idx_for_90_precision = (precisions >= 0.90).argmax()
    threshold_for_90_precision = thresholds[idx_for_90_precision]
    threshold_for_90_precision
    return idx_for_90_precision, threshold_for_90_precision


@app.cell
def _(threshold_for_90_precision, y_scores_1):
    y_train_pred_90 = y_scores_1 >= threshold_for_90_precision
    return (y_train_pred_90,)


@app.cell
def _(precision_score, y_train_5, y_train_pred_90):
    precision_score(y_train_5, y_train_pred_90)
    return


@app.cell
def _(recall_score, y_train_5, y_train_pred_90):
    recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
    recall_at_90_precision
    return (recall_at_90_precision,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The ROC Curve
        """
    )
    return


@app.cell
def _(y_scores_1, y_train_5):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds_1 = roc_curve(y_train_5, y_scores_1)
    return fpr, roc_curve, thresholds_1, tpr


@app.cell
def _(
    fpr,
    patches,
    plt,
    save_fig,
    threshold_for_90_precision,
    thresholds_1,
    tpr,
):
    idx_for_threshold_at_90 = (thresholds_1 <= threshold_for_90_precision).argmax()
    tpr_90, fpr_90 = (tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90])
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, linewidth=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
    plt.plot([fpr_90], [tpr_90], 'ko', label='Threshold for 90% precision')
    plt.gca().add_patch(patches.FancyArrowPatch((0.2, 0.89), (0.07, 0.7), connectionstyle='arc3,rad=.4', arrowstyle='Simple, tail_width=1.5, head_width=8, head_length=10', color='#444444'))
    plt.text(0.12, 0.71, 'Higher\nthreshold', color='#333333')
    plt.xlabel('False Positive Rate (Fall-Out)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.grid()
    plt.axis([0, 1, 0, 1])
    plt.legend(loc='lower right', fontsize=13)
    save_fig('roc_curve_plot')
    plt.show()
    return fpr_90, idx_for_threshold_at_90, tpr_90


@app.cell
def _(y_scores_1, y_train_5):
    from sklearn.metrics import roc_auc_score
    roc_auc_score(y_train_5, y_scores_1)
    return (roc_auc_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Warning:** the following cell may take a few minutes to run.
        """
    )
    return


@app.cell
def _():
    from sklearn.ensemble import RandomForestClassifier

    forest_clf = RandomForestClassifier(random_state=42)
    return RandomForestClassifier, forest_clf


@app.cell
def _(X_train, cross_val_predict, forest_clf, y_train_5):
    y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                        method="predict_proba")
    return (y_probas_forest,)


@app.cell
def _(y_probas_forest):
    y_probas_forest[:2]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        These are _estimated probabilities_. Among the images that the model classified as positive with a probability between 50% and 60%, there are actually about 94% positive images:
        """
    )
    return


@app.cell
def _(y_probas_forest, y_train_5):
    # Not in the code
    idx_50_to_60 = (y_probas_forest[:, 1] > 0.50) & (y_probas_forest[:, 1] < 0.60)
    print(f"{(y_train_5[idx_50_to_60]).sum() / idx_50_to_60.sum():.1%}")
    return (idx_50_to_60,)


@app.cell
def _(precision_recall_curve, y_probas_forest, y_train_5):
    y_scores_forest = y_probas_forest[:, 1]
    precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(
        y_train_5, y_scores_forest)
    return (
        precisions_forest,
        recalls_forest,
        thresholds_forest,
        y_scores_forest,
    )


@app.cell
def _(plt, precisions, precisions_forest, recalls, recalls_forest, save_fig):
    plt.figure(figsize=(6, 5))  # extra code – not needed, just formatting

    plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2,
             label="Random Forest")
    plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")

    # extra code – just beautifies and saves Figure 3–8
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.axis([0, 1, 0, 1])
    plt.grid()
    plt.legend(loc="lower left")
    save_fig("pr_curve_comparison_plot")

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We could use `cross_val_predict(forest_clf, X_train, y_train_5, cv=3)` to compute `y_train_pred_forest`, but since we already have the estimated probabilities, we can just use the default threshold of 50% probability to get the same predictions much faster:
        """
    )
    return


@app.cell
def _(f1_score, y_probas_forest, y_train_5):
    y_train_pred_forest = y_probas_forest[:, 1] >= 0.5  # positive proba ≥ 50%
    f1_score(y_train_5, y_train_pred_forest)
    return (y_train_pred_forest,)


@app.cell
def _(roc_auc_score, y_scores_forest, y_train_5):
    roc_auc_score(y_train_5, y_scores_forest)
    return


@app.cell
def _(precision_score, y_train_5, y_train_pred_forest):
    precision_score(y_train_5, y_train_pred_forest)
    return


@app.cell
def _(recall_score, y_train_5, y_train_pred_forest):
    recall_score(y_train_5, y_train_pred_forest)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Multiclass Classification
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        SVMs do not scale well to large datasets, so let's only train on the first 2,000 instances, or else this section will take a very long time to run:
        """
    )
    return


@app.cell
def _(X_train, y_train):
    from sklearn.svm import SVC

    svm_clf = SVC(random_state=42)
    svm_clf.fit(X_train[:2000], y_train[:2000])  # y_train, not y_train_5
    return SVC, svm_clf


@app.cell
def _(some_digit, svm_clf):
    svm_clf.predict([some_digit])
    return


@app.cell
def _(some_digit, svm_clf):
    some_digit_scores = svm_clf.decision_function([some_digit])
    some_digit_scores.round(2)
    return (some_digit_scores,)


@app.cell
def _(some_digit_scores):
    class_id = some_digit_scores.argmax()
    class_id
    return (class_id,)


@app.cell
def _(svm_clf):
    svm_clf.classes_
    return


@app.cell
def _(class_id, svm_clf):
    svm_clf.classes_[class_id]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If you want `decision_function()` to return all 45 scores, you can set the `decision_function_shape` hyperparameter to `"ovo"`. The default value is `"ovr"`, but don't let this confuse you: `SVC` always uses OvO for training. This hyperparameter only affects whether or not the 45 scores get aggregated or not:
        """
    )
    return


@app.cell
def _(some_digit, svm_clf):
    # extra code – shows how to get all 45 OvO scores if needed
    svm_clf.decision_function_shape = "ovo"
    some_digit_scores_ovo = svm_clf.decision_function([some_digit])
    some_digit_scores_ovo.round(2)
    return (some_digit_scores_ovo,)


@app.cell
def _(SVC, X_train, y_train):
    from sklearn.multiclass import OneVsRestClassifier

    ovr_clf = OneVsRestClassifier(SVC(random_state=42))
    ovr_clf.fit(X_train[:2000], y_train[:2000])
    return OneVsRestClassifier, ovr_clf


@app.cell
def _(ovr_clf, some_digit):
    ovr_clf.predict([some_digit])
    return


@app.cell
def _(ovr_clf):
    len(ovr_clf.estimators_)
    return


@app.cell
def _(SGDClassifier, X_train, some_digit, y_train):
    sgd_clf_1 = SGDClassifier(random_state=42)
    sgd_clf_1.fit(X_train, y_train)
    sgd_clf_1.predict([some_digit])
    return (sgd_clf_1,)


@app.cell
def _(sgd_clf_1, some_digit):
    sgd_clf_1.decision_function([some_digit]).round()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Warning:** the following two cells may take a few minutes each to run:
        """
    )
    return


@app.cell
def _(X_train, cross_val_score, sgd_clf_1, y_train):
    cross_val_score(sgd_clf_1, X_train, y_train, cv=3, scoring='accuracy')
    return


@app.cell
def _(X_train, cross_val_score, sgd_clf_1, y_train):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype('float64'))
    cross_val_score(sgd_clf_1, X_train_scaled, y_train, cv=3, scoring='accuracy')
    return StandardScaler, X_train_scaled, scaler


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Error Analysis
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Warning:** the following cell will take a few minutes to run:
        """
    )
    return


@app.cell
def _(X_train_scaled, cross_val_predict, plt, sgd_clf_1, y_train):
    from sklearn.metrics import ConfusionMatrixDisplay
    y_train_pred_1 = cross_val_predict(sgd_clf_1, X_train_scaled, y_train, cv=3)
    plt.rc('font', size=9)
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred_1)
    plt.show()
    return ConfusionMatrixDisplay, y_train_pred_1


@app.cell
def _(ConfusionMatrixDisplay, plt, y_train, y_train_pred_1):
    plt.rc('font', size=10)
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred_1, normalize='true', values_format='.0%')
    plt.show()
    return


@app.cell
def _(ConfusionMatrixDisplay, plt, y_train, y_train_pred_1):
    sample_weight = y_train_pred_1 != y_train
    plt.rc('font', size=10)
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred_1, sample_weight=sample_weight, normalize='true', values_format='.0%')
    plt.show()
    return (sample_weight,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's put all plots in a couple of figures for the book:
        """
    )
    return


@app.cell
def _(ConfusionMatrixDisplay, plt, save_fig, y_train, y_train_pred_1):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    plt.rc('font', size=9)
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred_1, ax=axs[0])
    axs[0].set_title('Confusion matrix')
    plt.rc('font', size=10)
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred_1, ax=axs[1], normalize='true', values_format='.0%')
    axs[1].set_title('CM normalized by row')
    save_fig('confusion_matrix_plot_1')
    plt.show()
    return axs, fig


@app.cell
def _(
    ConfusionMatrixDisplay,
    plt,
    sample_weight,
    save_fig,
    y_train,
    y_train_pred_1,
):
    fig_1, axs_1 = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
    plt.rc('font', size=10)
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred_1, ax=axs_1[0], sample_weight=sample_weight, normalize='true', values_format='.0%')
    axs_1[0].set_title('Errors normalized by row')
    ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred_1, ax=axs_1[1], sample_weight=sample_weight, normalize='pred', values_format='.0%')
    axs_1[1].set_title('Errors normalized by column')
    save_fig('confusion_matrix_plot_2')
    plt.show()
    plt.rc('font', size=14)
    return axs_1, fig_1


@app.cell
def _(X_train, y_train, y_train_pred_1):
    cl_a, cl_b = ('3', '5')
    X_aa = X_train[(y_train == cl_a) & (y_train_pred_1 == cl_a)]
    X_ab = X_train[(y_train == cl_a) & (y_train_pred_1 == cl_b)]
    X_ba = X_train[(y_train == cl_b) & (y_train_pred_1 == cl_a)]
    X_bb = X_train[(y_train == cl_b) & (y_train_pred_1 == cl_b)]
    return X_aa, X_ab, X_ba, X_bb, cl_a, cl_b


@app.cell
def _(X_aa, X_ab, X_ba, X_bb, cl_a, cl_b, plt, save_fig):
    size = 5
    pad = 0.2
    plt.figure(figsize=(size, size))
    for images, (label_col, label_row) in [(X_ba, (0, 0)), (X_bb, (1, 0)), (X_aa, (0, 1)), (X_ab, (1, 1))]:
        for idx_2, image_data_1 in enumerate(images[:size * size]):
            x = idx_2 % size + label_col * (size + pad)
            y_1 = idx_2 // size + label_row * (size + pad)
            plt.imshow(image_data_1.reshape(28, 28), cmap='binary', extent=(x, x + 1, y_1, y_1 + 1))
    plt.xticks([size / 2, size + pad + size / 2], [str(cl_a), str(cl_b)])
    plt.yticks([size / 2, size + pad + size / 2], [str(cl_b), str(cl_a)])
    plt.plot([size + pad / 2, size + pad / 2], [0, 2 * size + pad], 'k:')
    plt.plot([0, 2 * size + pad], [size + pad / 2, size + pad / 2], 'k:')
    plt.axis([0, 2 * size + pad, 0, 2 * size + pad])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    save_fig('error_analysis_digits_plot')
    plt.show()
    return idx_2, image_data_1, images, label_col, label_row, pad, size, x, y_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note: there are several other ways you could code a plot like this one, but it's a bit hard to get the axis labels right:
        * using [nested GridSpecs](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_nested.html)
        * merging all the digits in each block into a single image (then using 2×2 subplots). For example:
            ```python
            X_aa[:25].reshape(5, 5, 28, 28).transpose(0, 2, 1, 3).reshape(5 * 28, 5 * 28)
            ```
        * using [subfigures](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subfigures.html) (since Matplotlib 3.4)
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Multilabel Classification
        """
    )
    return


@app.cell
def _(X_train, y_train):
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier

    y_train_large = (y_train >= '7')
    y_train_odd = (y_train.astype('int8') % 2 == 1)
    y_multilabel = np.c_[y_train_large, y_train_odd]

    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_multilabel)
    return (
        KNeighborsClassifier,
        knn_clf,
        np,
        y_multilabel,
        y_train_large,
        y_train_odd,
    )


@app.cell
def _(knn_clf, some_digit):
    knn_clf.predict([some_digit])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Warning**: the following cell may take a few minutes to run:
        """
    )
    return


@app.cell
def _(X_train, cross_val_predict, f1_score, knn_clf, y_multilabel):
    y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
    f1_score(y_multilabel, y_train_knn_pred, average="macro")
    return (y_train_knn_pred,)


@app.cell
def _(f1_score, y_multilabel, y_train_knn_pred):
    # extra code – shows that we get a negligible performance improvement when we
    #              set average="weighted" because the classes are already pretty
    #              well balanced.
    f1_score(y_multilabel, y_train_knn_pred, average="weighted")
    return


@app.cell
def _(SVC, X_train, y_multilabel):
    from sklearn.multioutput import ClassifierChain

    chain_clf = ClassifierChain(SVC(), cv=3, random_state=42)
    chain_clf.fit(X_train[:2000], y_multilabel[:2000])
    return ClassifierChain, chain_clf


@app.cell
def _(chain_clf, some_digit):
    chain_clf.predict([some_digit])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Multioutput Classification
        """
    )
    return


@app.cell
def _(X_test, X_train, np):
    np.random.seed(42)  # to make this code example reproducible
    noise = np.random.randint(0, 100, (len(X_train), 784))
    X_train_mod = X_train + noise
    noise = np.random.randint(0, 100, (len(X_test), 784))
    X_test_mod = X_test + noise
    y_train_mod = X_train
    y_test_mod = X_test
    return X_test_mod, X_train_mod, noise, y_test_mod, y_train_mod


@app.cell
def _(X_test_mod, plot_digit, plt, save_fig, y_test_mod):
    # extra code – this cell generates and saves Figure 3–12
    plt.subplot(121); plot_digit(X_test_mod[0])
    plt.subplot(122); plot_digit(y_test_mod[0])
    save_fig("noisy_digit_example_plot")
    plt.show()
    return


@app.cell
def _(
    KNeighborsClassifier,
    X_test_mod,
    X_train_mod,
    plot_digit,
    plt,
    save_fig,
    y_train_mod,
):
    knn_clf_1 = KNeighborsClassifier()
    knn_clf_1.fit(X_train_mod, y_train_mod)
    clean_digit = knn_clf_1.predict([X_test_mod[0]])
    plot_digit(clean_digit)
    save_fig('cleaned_digit_example_plot')
    plt.show()
    return clean_digit, knn_clf_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Exercise solutions
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 1. An MNIST Classifier With Over 97% Accuracy
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Exercise: _Try to build a classifier for the MNIST dataset that achieves over 97% accuracy on the test set. Hint: the `KNeighborsClassifier` works quite well for this task; you just need to find good hyperparameter values (try a grid search on the `weights` and `n_neighbors` hyperparameters)._
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's start with a simple K-Nearest Neighbors classifier and measure its performance on the test set. This will be our baseline:
        """
    )
    return


@app.cell
def _(KNeighborsClassifier, X_test, X_train, y_test, y_train):
    knn_clf_2 = KNeighborsClassifier()
    knn_clf_2.fit(X_train, y_train)
    baseline_accuracy = knn_clf_2.score(X_test, y_test)
    baseline_accuracy
    return baseline_accuracy, knn_clf_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Great! A regular KNN classifier with the default hyperparameters is already very close to our goal.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's see if tuning the hyperparameters can help. To speed up the search, let's train only on the first 10,000 images:
        """
    )
    return


@app.cell
def _(KNeighborsClassifier, X_train, y_train):
    from sklearn.model_selection import GridSearchCV
    param_grid = [{'weights': ['uniform', 'distance'], 'n_neighbors': [3, 4, 5, 6]}]
    knn_clf_3 = KNeighborsClassifier()
    grid_search = GridSearchCV(knn_clf_3, param_grid, cv=5)
    grid_search.fit(X_train[:10000], y_train[:10000])
    return GridSearchCV, grid_search, knn_clf_3, param_grid


@app.cell
def _(grid_search):
    grid_search.best_params_
    return


@app.cell
def _(grid_search):
    grid_search.best_score_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The score dropped, but that was expected since we only trained on 10,000 images. So let's take the best model and train it again on the full training set:
        """
    )
    return


@app.cell
def _(X_test, X_train, grid_search, y_test, y_train):
    grid_search.best_estimator_.fit(X_train, y_train)
    tuned_accuracy = grid_search.score(X_test, y_test)
    tuned_accuracy
    return (tuned_accuracy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We reached our goal of 97% accuracy! 🥳
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Data Augmentation
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Exercise: _Write a function that can shift an MNIST image in any direction (left, right, up, or down) by one pixel. You can use the `shift()` function from the `scipy.ndimage` module. For example, `shift(image, [2, 1], cval=0)` shifts the image two pixels down and one pixel to the right. Then, for each image in the training set, create four shifted copies (one per direction) and add them to the training set. Finally, train your best model on this expanded training set and measure its accuracy on the test set. You should observe that your model performs even better now! This technique of artificially growing the training set is called _data augmentation_ or _training set expansion_._
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's try augmenting the MNIST dataset by adding slightly shifted versions of each image.
        """
    )
    return


@app.cell
def _():
    from scipy.ndimage import shift
    return (shift,)


@app.cell
def _(shift):
    def shift_image(image, dx, dy):
        image = image.reshape((28, 28))
        shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
        return shifted_image.reshape([-1])
    return (shift_image,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's see if it works:
        """
    )
    return


@app.cell
def _(X_train, plt, shift_image):
    image = X_train[1000]  # some random digit to demo
    shifted_image_down = shift_image(image, 0, 5)
    shifted_image_left = shift_image(image, -5, 0)

    plt.figure(figsize=(12, 3))
    plt.subplot(131)
    plt.title("Original")
    plt.imshow(image.reshape(28, 28),
               interpolation="nearest", cmap="Greys")
    plt.subplot(132)
    plt.title("Shifted down")
    plt.imshow(shifted_image_down.reshape(28, 28),
               interpolation="nearest", cmap="Greys")
    plt.subplot(133)
    plt.title("Shifted left")
    plt.imshow(shifted_image_left.reshape(28, 28),
               interpolation="nearest", cmap="Greys")
    plt.show()
    return image, shifted_image_down, shifted_image_left


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Looks good! Now let's create an augmented training set by shifting every image left, right, up and down by one pixel:
        """
    )
    return


@app.cell
def _(X_train, np, shift_image, y_train):
    X_train_augmented = [image for image in X_train]
    y_train_augmented = [label for label in y_train]
    for dx, dy in ((-1, 0), (1, 0), (0, 1), (0, -1)):
        for image_1, label in zip(X_train, y_train):
            X_train_augmented.append(shift_image(image_1, dx, dy))
            y_train_augmented.append(label)
    X_train_augmented = np.array(X_train_augmented)
    y_train_augmented = np.array(y_train_augmented)
    return X_train_augmented, dx, dy, image_1, label, y_train_augmented


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's shuffle the augmented training set, or else all shifted images will be grouped together:
        """
    )
    return


@app.cell
def _(X_train_augmented, np, y_train_augmented):
    shuffle_idx = np.random.permutation(len(X_train_augmented))
    X_train_augmented_1 = X_train_augmented[shuffle_idx]
    y_train_augmented_1 = y_train_augmented[shuffle_idx]
    return X_train_augmented_1, shuffle_idx, y_train_augmented_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's train the model using the best hyperparameters we found in the previous exercise:
        """
    )
    return


@app.cell
def _(KNeighborsClassifier, grid_search):
    knn_clf_4 = KNeighborsClassifier(**grid_search.best_params_)
    return (knn_clf_4,)


@app.cell
def _(X_train_augmented_1, knn_clf_4, y_train_augmented_1):
    knn_clf_4.fit(X_train_augmented_1, y_train_augmented_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Warning**: the following cell may take a few minutes to run:
        """
    )
    return


@app.cell
def _(X_test, knn_clf_4, y_test):
    augmented_accuracy = knn_clf_4.score(X_test, y_test)
    augmented_accuracy
    return (augmented_accuracy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        By simply augmenting the data, we've got a 0.5% accuracy boost. Perhaps it does not sound so impressive, but it actually means that the error rate dropped significantly:
        """
    )
    return


@app.cell
def _(augmented_accuracy, tuned_accuracy):
    error_rate_change = (1 - augmented_accuracy) / (1 - tuned_accuracy) - 1
    print(f"error_rate_change = {error_rate_change:.0%}")
    return (error_rate_change,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The error rate dropped quite a bit thanks to data augmentation.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3. Tackle the Titanic dataset
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Exercise: _Tackle the Titanic dataset. A great place to start is on [Kaggle](https://www.kaggle.com/c/titanic). Alternatively, you can download the data from https://homl.info/titanic.tgz and unzip this tarball like you did for the housing data in Chapter 2. This will give you two CSV files: _train.csv_ and _test.csv_ which you can load using `pandas.read_csv()`. The goal is to train a classifier that can predict the `Survived` column based on the other columns._
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's fetch the data and load it:
        """
    )
    return


@app.cell
def _(Path):
    import pandas as pd
    import tarfile
    import urllib.request

    def load_titanic_data():
        tarball_path = Path("datasets/titanic.tgz")
        if not tarball_path.is_file():
            Path("datasets").mkdir(parents=True, exist_ok=True)
            url = "https://github.com/ageron/data/raw/main/titanic.tgz"
            urllib.request.urlretrieve(url, tarball_path)
            with tarfile.open(tarball_path) as titanic_tarball:
                titanic_tarball.extractall(path="datasets")
        return [pd.read_csv(Path("datasets/titanic") / filename)
                for filename in ("train.csv", "test.csv")]
    return load_titanic_data, pd, tarfile, urllib


@app.cell
def _(load_titanic_data):
    train_data, test_data = load_titanic_data()
    return test_data, train_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The data is already split into a training set and a test set. However, the test data does *not* contain the labels: your goal is to train the best model you can on the training data, then make your predictions on the test data and upload them to Kaggle to see your final score.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's take a peek at the top few rows of the training set:
        """
    )
    return


@app.cell
def _(train_data):
    train_data.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The attributes have the following meaning:
        * **PassengerId**: a unique identifier for each passenger
        * **Survived**: that's the target, 0 means the passenger did not survive, while 1 means he/she survived.
        * **Pclass**: passenger class.
        * **Name**, **Sex**, **Age**: self-explanatory
        * **SibSp**: how many siblings & spouses of the passenger aboard the Titanic.
        * **Parch**: how many children & parents of the passenger aboard the Titanic.
        * **Ticket**: ticket id
        * **Fare**: price paid (in pounds)
        * **Cabin**: passenger's cabin number
        * **Embarked**: where the passenger embarked the Titanic
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The goal is to predict whether or not a passenger survived based on attributes such as their age, sex, passenger class, where they embarked and so on.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's explicitly set the `PassengerId` column as the index column:
        """
    )
    return


@app.cell
def _(test_data, train_data):
    train_data_1 = train_data.set_index('PassengerId')
    test_data_1 = test_data.set_index('PassengerId')
    return test_data_1, train_data_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's get more info to see how much data is missing:
        """
    )
    return


@app.cell
def _(train_data_1):
    train_data_1.info()
    return


@app.cell
def _(train_data_1):
    train_data_1[train_data_1['Sex'] == 'female']['Age'].median()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Okay, the **Age**, **Cabin** and **Embarked** attributes are sometimes null (less than 891 non-null), especially the **Cabin** (77% are null). We will ignore the **Cabin** for now and focus on the rest. The **Age** attribute has about 19% null values, so we will need to decide what to do with them. Replacing null values with the median age seems reasonable. We could be a bit smarter by predicting the age based on the other columns (for example, the median age is 37 in 1st class, 29 in 2nd class and 24 in 3rd class), but we'll keep things simple and just use the overall median age.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The **Name** and **Ticket** attributes may have some value, but they will be a bit tricky to convert into useful numbers that a model can consume. So for now, we will ignore them.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's take a look at the numerical attributes:
        """
    )
    return


@app.cell
def _(train_data_1):
    train_data_1.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        * Yikes, only 38% **Survived**! 😭 That's close enough to 40%, so accuracy will be a reasonable metric to evaluate our model.
        * The mean **Fare** was £32.20, which does not seem so expensive (but it was probably a lot of money back then).
        * The mean **Age** was less than 30 years old.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's check that the target is indeed 0 or 1:
        """
    )
    return


@app.cell
def _(train_data_1):
    train_data_1['Survived'].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's take a quick look at all the categorical attributes:
        """
    )
    return


@app.cell
def _(train_data_1):
    train_data_1['Pclass'].value_counts()
    return


@app.cell
def _(train_data_1):
    train_data_1['Sex'].value_counts()
    return


@app.cell
def _(train_data_1):
    train_data_1['Embarked'].value_counts()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The Embarked attribute tells us where the passenger embarked: C=Cherbourg, Q=Queenstown, S=Southampton.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's build our preprocessing pipelines, starting with the pipeline for numerical attributes:
        """
    )
    return


@app.cell
def _(StandardScaler):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
    return Pipeline, SimpleImputer, num_pipeline


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now we can build the pipeline for the categorical attributes:
        """
    )
    return


@app.cell
def _():
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
    return OneHotEncoder, OrdinalEncoder


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note: the `sparse` hyperparameter below was renamed to `sparse_output`.
        """
    )
    return


@app.cell
def _(OneHotEncoder, OrdinalEncoder, Pipeline, SimpleImputer):
    cat_pipeline = Pipeline([
            ("ordinal_encoder", OrdinalEncoder()),    
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("cat_encoder", OneHotEncoder(sparse_output=False)),
        ])
    return (cat_pipeline,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Finally, let's join the numerical and categorical pipelines:
        """
    )
    return


@app.cell
def _(cat_pipeline, num_pipeline):
    from sklearn.compose import ColumnTransformer

    num_attribs = ["Age", "SibSp", "Parch", "Fare"]
    cat_attribs = ["Pclass", "Sex", "Embarked"]

    preprocess_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", cat_pipeline, cat_attribs),
        ])
    return ColumnTransformer, cat_attribs, num_attribs, preprocess_pipeline


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Cool! Now we have a nice preprocessing pipeline that takes the raw data and outputs numerical input features that we can feed to any Machine Learning model we want.
        """
    )
    return


@app.cell
def _(preprocess_pipeline, train_data_1):
    X_train_1 = preprocess_pipeline.fit_transform(train_data_1)
    X_train_1
    return (X_train_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's not forget to get the labels:
        """
    )
    return


@app.cell
def _(train_data_1):
    y_train_1 = train_data_1['Survived']
    return (y_train_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We are now ready to train a classifier. Let's start with a `RandomForestClassifier`:
        """
    )
    return


@app.cell
def _(RandomForestClassifier, X_train_1, y_train_1):
    forest_clf_1 = RandomForestClassifier(n_estimators=100, random_state=42)
    forest_clf_1.fit(X_train_1, y_train_1)
    return (forest_clf_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Great, our model is trained, let's use it to make predictions on the test set:
        """
    )
    return


@app.cell
def _(forest_clf_1, preprocess_pipeline, test_data_1):
    X_test_1 = preprocess_pipeline.transform(test_data_1)
    y_pred_1 = forest_clf_1.predict(X_test_1)
    return X_test_1, y_pred_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And now we could just build a CSV file with these predictions (respecting the format expected by Kaggle), then upload it and hope for the best. But wait! We can do better than hope. Why don't we use cross-validation to have an idea of how good our model is?
        """
    )
    return


@app.cell
def _(X_train_1, cross_val_score, forest_clf_1, y_train_1):
    forest_scores = cross_val_score(forest_clf_1, X_train_1, y_train_1, cv=10)
    forest_scores.mean()
    return (forest_scores,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Okay, not too bad! Looking at the [leaderboard](https://www.kaggle.com/c/titanic/leaderboard) for the Titanic competition on Kaggle, you can see that our score is in the top 2%, woohoo! Some Kagglers reached 100% accuracy, but since you can easily find the [list of victims](https://www.encyclopedia-titanica.org/titanic-victims/) of the Titanic, it seems likely that there was little Machine Learning involved in their performance! 😆
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's try an `SVC`:
        """
    )
    return


@app.cell
def _(SVC, X_train_1, cross_val_score, y_train_1):
    svm_clf_1 = SVC(gamma='auto')
    svm_scores = cross_val_score(svm_clf_1, X_train_1, y_train_1, cv=10)
    svm_scores.mean()
    return svm_clf_1, svm_scores


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Great! This model looks better.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        But instead of just looking at the mean accuracy across the 10 cross-validation folds, let's plot all 10 scores for each model, along with a box plot highlighting the lower and upper quartiles, and "whiskers" showing the extent of the scores (thanks to Nevin Yilmaz for suggesting this visualization). Note that the `boxplot()` function detects outliers (called "fliers") and does not include them within the whiskers. Specifically, if the lower quartile is $Q_1$ and the upper quartile is $Q_3$, then the interquartile range $IQR = Q_3 - Q_1$ (this is the box's height), and any score lower than $Q_1 - 1.5 \times IQR$ is a flier, and so is any score greater than $Q3 + 1.5 \times IQR$.
        """
    )
    return


@app.cell
def _(forest_scores, plt, svm_scores):
    plt.figure(figsize=(8, 4))
    plt.plot([1]*10, svm_scores, ".")
    plt.plot([2]*10, forest_scores, ".")
    plt.boxplot([svm_scores, forest_scores], labels=("SVM", "Random Forest"))
    plt.ylabel("Accuracy")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The random forest classifier got a very high score on one of the 10 folds, but overall it had a lower mean score, as well as a bigger spread, so it looks like the SVM classifier is more likely to generalize well.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To improve this result further, you could:
        * Compare many more models and tune hyperparameters using cross validation and grid search,
        * Do more feature engineering, for example:
          * Try to convert numerical attributes to categorical attributes: for example, different age groups had very different survival rates (see below), so it may help to create an age bucket category and use it instead of the age. Similarly, it may be useful to have a special category for people traveling alone since only 30% of them survived (see below).
          * Replace **SibSp** and **Parch** with their sum.
          * Try to identify parts of names that correlate well with the **Survived** attribute.
          * Use the **Cabin** column, for example take its first letter and treat it as a categorical attribute.
        """
    )
    return


@app.cell
def _(train_data_1):
    train_data_1['AgeBucket'] = train_data_1['Age'] // 15 * 15
    train_data_1[['AgeBucket', 'Survived']].groupby(['AgeBucket']).mean()
    return


@app.cell
def _(train_data_1):
    train_data_1['RelativesOnboard'] = train_data_1['SibSp'] + train_data_1['Parch']
    train_data_1[['RelativesOnboard', 'Survived']].groupby(['RelativesOnboard']).mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Spam classifier
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Exercise: _Build a spam classifier (a more challenging exercise):_

        * _Download examples of spam and ham from [Apache SpamAssassin's public datasets](https://homl.info/spamassassin)._
        * _Unzip the datasets and familiarize yourself with the data format._
        * _Split the datasets into a training set and a test set._
        * _Write a data preparation pipeline to convert each email into a feature vector. Your preparation pipeline should transform an email into a (sparse) vector that indicates the presence or absence of each possible word. For example, if all emails only ever contain four words, "Hello," "how," "are," "you," then the email "Hello you Hello Hello you" would be converted into a vector [1, 0, 0, 1] (meaning [“Hello" is present, "how" is absent, "are" is absent, "you" is present]), or [3, 0, 0, 2] if you prefer to count the number of occurrences of each word._

        _You may want to add hyperparameters to your preparation pipeline to control whether or not to strip off email headers, convert each email to lowercase, remove punctuation, replace all URLs with "URL," replace all numbers with "NUMBER," or even perform _stemming_ (i.e., trim off word endings; there are Python libraries available to do this)._

        _Finally, try out several classifiers and see if you can build a great spam classifier, with both high recall and high precision._
        """
    )
    return


@app.cell
def _(Path, tarfile, urllib):
    def fetch_spam_data():
        spam_root = "http://spamassassin.apache.org/old/publiccorpus/"
        ham_url = spam_root + "20030228_easy_ham.tar.bz2"
        spam_url = spam_root + "20030228_spam.tar.bz2"

        spam_path = Path() / "datasets" / "spam"
        spam_path.mkdir(parents=True, exist_ok=True)
        for dir_name, tar_name, url in (("easy_ham", "ham", ham_url),
                                        ("spam", "spam", spam_url)):
            if not (spam_path / dir_name).is_dir():
                path = (spam_path / tar_name).with_suffix(".tar.bz2")
                print("Downloading", path)
                urllib.request.urlretrieve(url, path)
                tar_bz2_file = tarfile.open(path)
                tar_bz2_file.extractall(path=spam_path)
                tar_bz2_file.close()
        return [spam_path / dir_name for dir_name in ("easy_ham", "spam")]
    return (fetch_spam_data,)


@app.cell
def _(fetch_spam_data):
    ham_dir, spam_dir = fetch_spam_data()
    return ham_dir, spam_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Next, let's load all the emails:
        """
    )
    return


@app.cell
def _(ham_dir, spam_dir):
    ham_filenames = [f for f in sorted(ham_dir.iterdir()) if len(f.name) > 20]
    spam_filenames = [f for f in sorted(spam_dir.iterdir()) if len(f.name) > 20]
    return ham_filenames, spam_filenames


@app.cell
def _(ham_filenames):
    len(ham_filenames)
    return


@app.cell
def _(spam_filenames):
    len(spam_filenames)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can use Python's `email` module to parse these emails (this handles headers, encoding, and so on):
        """
    )
    return


@app.cell
def _():
    import email
    import email.policy

    def load_email(filepath):
        with open(filepath, "rb") as f:
            return email.parser.BytesParser(policy=email.policy.default).parse(f)
    return email, load_email


@app.cell
def _(ham_filenames, load_email, spam_filenames):
    ham_emails = [load_email(filepath) for filepath in ham_filenames]
    spam_emails = [load_email(filepath) for filepath in spam_filenames]
    return ham_emails, spam_emails


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's look at one example of ham and one example of spam, to get a feel of what the data looks like:
        """
    )
    return


@app.cell
def _(ham_emails):
    print(ham_emails[1].get_content().strip())
    return


@app.cell
def _(spam_emails):
    print(spam_emails[6].get_content().strip())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Some emails are actually multipart, with images and attachments (which can have their own attachments). Let's look at the various types of structures we have:
        """
    )
    return


@app.cell
def _():
    def get_email_structure(email):
        if isinstance(email, str):
            return email
        payload = email.get_payload()
        if isinstance(payload, list):
            multipart = ", ".join([get_email_structure(sub_email)
                                   for sub_email in payload])
            return f"multipart({multipart})"
        else:
            return email.get_content_type()
    return (get_email_structure,)


@app.cell
def _(get_email_structure):
    from collections import Counter

    def structures_counter(emails):
        structures = Counter()
        for email in emails:
            structure = get_email_structure(email)
            structures[structure] = structures[structure] + 1
        return structures
    return Counter, structures_counter


@app.cell
def _(ham_emails, structures_counter):
    structures_counter(ham_emails).most_common()
    return


@app.cell
def _(spam_emails, structures_counter):
    structures_counter(spam_emails).most_common()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It seems that the ham emails are more often plain text, while spam has quite a lot of HTML. Moreover, quite a few ham emails are signed using PGP, while no spam is. In short, it seems that the email structure is useful information to have.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's take a look at the email headers:
        """
    )
    return


@app.cell
def _(spam_emails):
    for header, value in spam_emails[0].items():
        print(header, ":", value)
    return header, value


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        There's probably a lot of useful information in there, such as the sender's email address (12a1mailbot1@web.de looks fishy), but we will just focus on the `Subject` header:
        """
    )
    return


@app.cell
def _(spam_emails):
    spam_emails[0]["Subject"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Okay, before we learn too much about the data, let's not forget to split it into a training set and a test set:
        """
    )
    return


@app.cell
def _(ham_emails, np, spam_emails):
    from sklearn.model_selection import train_test_split
    X_1 = np.array(ham_emails + spam_emails, dtype=object)
    y_2 = np.array([0] * len(ham_emails) + [1] * len(spam_emails))
    X_train_2, X_test_2, y_train_2, y_test_1 = train_test_split(X_1, y_2, test_size=0.2, random_state=42)
    return X_1, X_test_2, X_train_2, train_test_split, y_2, y_test_1, y_train_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Okay, let's start writing the preprocessing functions. First, we will need a function to convert HTML to plain text. Arguably the best way to do this would be to use the great [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) library, but I would like to avoid adding another dependency to this project, so let's hack a quick & dirty solution using regular expressions (at the risk of [un̨ho͞ly radiańcé destro҉ying all enli̍̈́̂̈́ghtenment](https://stackoverflow.com/a/1732454/38626)). The following function first drops the `<head>` section, then converts all `<a>` tags to the word HYPERLINK, then it gets rid of all HTML tags, leaving only the plain text. For readability, it also replaces multiple newlines with single newlines, and finally it unescapes html entities (such as `&gt;` or `&nbsp;`):
        """
    )
    return


@app.cell
def _():
    import re
    from html import unescape

    def html_to_plain_text(html):
        text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
        text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
        text = re.sub('<.*?>', '', text, flags=re.M | re.S)
        text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
        return unescape(text)
    return html_to_plain_text, re, unescape


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's see if it works. This is HTML spam:
        """
    )
    return


@app.cell
def _(X_train_2, get_email_structure, y_train_2):
    html_spam_emails = [email for email in X_train_2[y_train_2 == 1] if get_email_structure(email) == 'text/html']
    sample_html_spam = html_spam_emails[7]
    print(sample_html_spam.get_content().strip()[:1000], '...')
    return html_spam_emails, sample_html_spam


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And this is the resulting plain text:
        """
    )
    return


@app.cell
def _(html_to_plain_text, sample_html_spam):
    print(html_to_plain_text(sample_html_spam.get_content())[:1000], "...")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Great! Now let's write a function that takes an email as input and returns its content as plain text, whatever its format is:
        """
    )
    return


@app.cell
def _(html_to_plain_text):
    def email_to_text(email):
        html = None
        for part in email.walk():
            ctype = part.get_content_type()
            if not ctype in ("text/plain", "text/html"):
                continue
            try:
                content = part.get_content()
            except: # in case of encoding issues
                content = str(part.get_payload())
            if ctype == "text/plain":
                return content
            else:
                html = content
        if html:
            return html_to_plain_text(html)
    return (email_to_text,)


@app.cell
def _(email_to_text, sample_html_spam):
    print(email_to_text(sample_html_spam)[:100], "...")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's throw in some stemming! We will use the Natural Language Toolkit ([NLTK](http://www.nltk.org/)):
        """
    )
    return


@app.cell
def _():
    import nltk

    stemmer = nltk.PorterStemmer()
    for word in ("Computations", "Computation", "Computing", "Computed", "Compute",
                 "Compulsive"):
        print(word, "=>", stemmer.stem(word))
    return nltk, stemmer, word


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We will also need a way to replace URLs with the word "URL". For this, we could use hard core [regular expressions](https://mathiasbynens.be/demo/url-regex) but we will just use the [urlextract](https://github.com/lipoja/URLExtract) library:
        """
    )
    return


app._unparsable_cell(
    r"""
    # Is this notebook running on Colab or Kaggle?
    IS_COLAB = \"google.colab\" in sys.modules
    IS_KAGGLE = \"kaggle_secrets\" in sys.modules

    # if running this notebook on Colab or Kaggle, we just pip install urlextract
    if IS_COLAB or IS_KAGGLE:
        %pip install -q -U urlextract
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Note:** inside a Jupyter notebook, always use `%pip` instead of `!pip`, as `!pip` may install the library inside the wrong environment, while `%pip` makes sure it's installed inside the currently running environment.
        """
    )
    return


@app.cell
def _():
    import urlextract # may require an Internet connection to download root domain
                      # names

    url_extractor = urlextract.URLExtract()
    some_text = "Will it detect github.com and https://youtu.be/7Pq-S557XQU?t=3m32s"
    print(url_extractor.find_urls(some_text))
    return some_text, url_extractor, urlextract


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We are ready to put all this together into a transformer that we will use to convert emails to word counters. Note that we split sentences into words using Python's `split()` method, which uses whitespaces for word boundaries. This works for many written languages, but not all. For example, Chinese and Japanese scripts generally don't use spaces between words, and Vietnamese often uses spaces even between syllables. It's okay in this exercise, because the dataset is (mostly) in English.
        """
    )
    return


@app.cell
def _(Counter, email_to_text, np, re, stemmer, url_extractor):
    from sklearn.base import BaseEstimator, TransformerMixin

    class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):

        def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True, replace_urls=True, replace_numbers=True, stemming=True):
            self.strip_headers = strip_headers
            self.lower_case = lower_case
            self.remove_punctuation = remove_punctuation
            self.replace_urls = replace_urls
            self.replace_numbers = replace_numbers
            self.stemming = stemming

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            X_transformed = []
            for email in X:
                text = email_to_text(email) or ''
                if self.lower_case:
                    text = text.lower()
                if self.replace_urls and url_extractor is not None:
                    urls = list(set(url_extractor.find_urls(text)))
                    urls.sort(key=lambda url: len(url), reverse=True)
                    for url in urls:
                        text = text.replace(url, ' URL ')
                if self.replace_numbers:
                    text = re.sub('\\d+(?:\\.\\d*)?(?:[eE][+-]?\\d+)?', 'NUMBER', text)
                if self.remove_punctuation:
                    text = re.sub('\\W+', ' ', text, flags=re.M)
                word_counts = Counter(text.split())
                if self.stemming and stemmer is not None:
                    stemmed_word_counts = Counter()
                    for word, count in word_counts.items():
                        stemmed_word = stemmer.stem(word)
                        stemmed_word_counts[stemmed_word] = stemmed_word_counts[stemmed_word] + count
                    word_counts = stemmed_word_counts
                X_transformed.append(word_counts)
            return np.array(X_transformed)
    return BaseEstimator, EmailToWordCounterTransformer, TransformerMixin


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's try this transformer on a few emails:
        """
    )
    return


@app.cell
def _(EmailToWordCounterTransformer, X_train_2):
    X_few = X_train_2[:3]
    X_few_wordcounts = EmailToWordCounterTransformer().fit_transform(X_few)
    X_few_wordcounts
    return X_few, X_few_wordcounts


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This looks about right!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now we have the word counts, and we need to convert them to vectors. For this, we will build another transformer whose `fit()` method will build the vocabulary (an ordered list of the most common words) and whose `transform()` method will use the vocabulary to convert word counts to vectors. The output is a sparse matrix.
        """
    )
    return


@app.cell
def _(BaseEstimator, Counter, TransformerMixin):
    from scipy.sparse import csr_matrix

    class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):

        def __init__(self, vocabulary_size=1000):
            self.vocabulary_size = vocabulary_size

        def fit(self, X, y=None):
            total_count = Counter()
            for word_count in X:
                for word, count in word_count.items():
                    total_count[word] = total_count[word] + min(count, 10)
            most_common = total_count.most_common()[:self.vocabulary_size]
            self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
            return self

        def transform(self, X, y=None):
            rows = []
            cols = []
            data = []
            for row, word_count in enumerate(X):
                for word, count in word_count.items():
                    rows.append(row)
                    cols.append(self.vocabulary_.get(word, 0))
                    data.append(count)
            return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))
    return WordCounterToVectorTransformer, csr_matrix


@app.cell
def _(WordCounterToVectorTransformer, X_few_wordcounts):
    vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=10)
    X_few_vectors = vocab_transformer.fit_transform(X_few_wordcounts)
    X_few_vectors
    return X_few_vectors, vocab_transformer


@app.cell
def _(X_few_vectors):
    X_few_vectors.toarray()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        What does this matrix mean? Well, the 99 in the second row, first column, means that the second email contains 99 words that are not part of the vocabulary. The 11 next to it means that the first word in the vocabulary is present 11 times in this email. The 9 next to it means that the second word is present 9 times, and so on. You can look at the vocabulary to know which words we are talking about. The first word is "the", the second word is "of", etc.
        """
    )
    return


@app.cell
def _(vocab_transformer):
    vocab_transformer.vocabulary_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We are now ready to train our first spam classifier! Let's transform the whole dataset:
        """
    )
    return


@app.cell
def _(
    EmailToWordCounterTransformer,
    Pipeline,
    WordCounterToVectorTransformer,
    X_train_2,
):
    preprocess_pipeline_1 = Pipeline([('email_to_wordcount', EmailToWordCounterTransformer()), ('wordcount_to_vector', WordCounterToVectorTransformer())])
    X_train_transformed = preprocess_pipeline_1.fit_transform(X_train_2)
    return X_train_transformed, preprocess_pipeline_1


@app.cell
def _(X_train_transformed, cross_val_score, y_train_2):
    from sklearn.linear_model import LogisticRegression
    log_clf = LogisticRegression(max_iter=1000, random_state=42)
    score = cross_val_score(log_clf, X_train_transformed, y_train_2, cv=3)
    score.mean()
    return LogisticRegression, log_clf, score


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Over 98.5%, not bad for a first try! :) However, remember that we are using the "easy" dataset. You can try with the harder datasets, the results won't be so amazing. You would have to try multiple models, select the best ones and fine-tune them using cross-validation, and so on.

        But you get the picture, so let's stop now, and just print out the precision/recall we get on the test set:
        """
    )
    return


@app.cell
def _(
    LogisticRegression,
    X_test_2,
    X_train_transformed,
    precision_score,
    preprocess_pipeline_1,
    recall_score,
    y_test_1,
    y_train_2,
):
    X_test_transformed = preprocess_pipeline_1.transform(X_test_2)
    log_clf_1 = LogisticRegression(max_iter=1000, random_state=42)
    log_clf_1.fit(X_train_transformed, y_train_2)
    y_pred_2 = log_clf_1.predict(X_test_transformed)
    print(f'Precision: {precision_score(y_test_1, y_pred_2):.2%}')
    print(f'Recall: {recall_score(y_test_1, y_pred_2):.2%}')
    return X_test_transformed, log_clf_1, y_pred_2


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

