import marimo

__generated_with = "0.11.20"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Chapter 7 – Ensemble Learning and Random Forests**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        _This notebook contains all the sample code and solutions to the exercises in chapter 7._
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <table align="left">
          <td>
            <a href="https://colab.research.google.com/github/ageron/handson-ml3/blob/main/07_ensemble_learning_and_random_forests.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
          </td>
          <td>
            <a target="_blank" href="https://kaggle.com/kernels/welcome?src=https://github.com/ageron/handson-ml3/blob/main/07_ensemble_learning_and_random_forests.ipynb"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>
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
        As we did in previous chapters, let's define the default font sizes to make the figures prettier:
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
        And let's create the `images/ensembles` folder (if it doesn't already exist), and define the `save_fig()` function which is used through this notebook to save the figures in high-res for the book:
        """
    )
    return


@app.cell
def _(plt):
    from pathlib import Path

    IMAGES_PATH = Path() / "images" / "ensembles"
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
        # Voting Classifiers
        """
    )
    return


@app.cell
def _(plt, save_fig):
    # extra code – this cell generates and saves Figure 7–3

    import numpy as np

    heads_proba = 0.51
    np.random.seed(42)
    coin_tosses = (np.random.rand(10000, 10) < heads_proba).astype(np.int32)
    cumulative_heads = coin_tosses.cumsum(axis=0)
    cumulative_heads_ratio = cumulative_heads / np.arange(1, 10001).reshape(-1, 1)

    plt.figure(figsize=(8, 3.5))
    plt.plot(cumulative_heads_ratio)
    plt.plot([0, 10000], [0.51, 0.51], "k--", linewidth=2, label="51%")
    plt.plot([0, 10000], [0.5, 0.5], "k-", label="50%")
    plt.xlabel("Number of coin tosses")
    plt.ylabel("Heads ratio")
    plt.legend(loc="lower right")
    plt.axis([0, 10000, 0.42, 0.58])
    plt.grid()
    save_fig("law_of_large_numbers_plot")
    plt.show()
    return (
        coin_tosses,
        cumulative_heads,
        cumulative_heads_ratio,
        heads_proba,
        np,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's build a voting classifier:
        """
    )
    return


@app.cell
def _():
    from sklearn.datasets import make_moons
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    voting_clf = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(random_state=42)),
            ('rf', RandomForestClassifier(random_state=42)),
            ('svc', SVC(random_state=42))
        ]
    )
    voting_clf.fit(X_train, y_train)
    return (
        LogisticRegression,
        RandomForestClassifier,
        SVC,
        VotingClassifier,
        X,
        X_test,
        X_train,
        make_moons,
        train_test_split,
        voting_clf,
        y,
        y_test,
        y_train,
    )


@app.cell
def _(X_test, voting_clf, y_test):
    for _name, clf in voting_clf.named_estimators_.items():
        print(_name, '=', clf.score(X_test, y_test))
    return (clf,)


@app.cell
def _(X_test, voting_clf):
    voting_clf.predict(X_test[:1])
    return


@app.cell
def _(X_test, voting_clf):
    [clf.predict(X_test[:1]) for clf in voting_clf.estimators_]
    return


@app.cell
def _(X_test, voting_clf, y_test):
    voting_clf.score(X_test, y_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's use soft voting:
        """
    )
    return


@app.cell
def _(X_test, X_train, voting_clf, y_test, y_train):
    voting_clf.voting = "soft"
    voting_clf.named_estimators["svc"].probability = True
    voting_clf.fit(X_train, y_train)
    voting_clf.score(X_test, y_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Bagging and Pasting
        ## Bagging and Pasting in Scikit-Learn
        """
    )
    return


@app.cell
def _(X_train, y_train):
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier

    bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
                                max_samples=100, n_jobs=-1, random_state=42)
    bag_clf.fit(X_train, y_train)
    return BaggingClassifier, DecisionTreeClassifier, bag_clf


@app.cell
def _(DecisionTreeClassifier, X_train, bag_clf, np, plt, save_fig, y_train):
    def plot_decision_boundary(clf, X, y, alpha=1.0):
        _axes = [-1.5, 2.4, -1, 1.5]
        x1, x2 = np.meshgrid(np.linspace(_axes[0], _axes[1], 100), np.linspace(_axes[2], _axes[3], 100))
        X_new = np.c_[x1.ravel(), x2.ravel()]
        y_pred = clf.predict(X_new).reshape(x1.shape)
        plt.contourf(x1, x2, y_pred, alpha=0.3 * alpha, cmap='Wistia')
        plt.contour(x1, x2, y_pred, cmap='Greys', alpha=0.8 * alpha)
        colors = ['#78785c', '#c47b27']
        markers = ('o', '^')
        for idx in (0, 1):
            plt.plot(X[:, 0][y == idx], X[:, 1][y == idx], color=colors[idx], marker=markers[idx], linestyle='none')
        plt.axis(_axes)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$', rotation=0)
    tree_clf = DecisionTreeClassifier(random_state=42)
    tree_clf.fit(X_train, y_train)
    _fig, _axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
    plt.sca(_axes[0])
    plot_decision_boundary(tree_clf, X_train, y_train)
    plt.title('Decision Tree')
    plt.sca(_axes[1])
    plot_decision_boundary(bag_clf, X_train, y_train)
    plt.title('Decision Trees with Bagging')
    plt.ylabel('')
    save_fig('decision_tree_without_and_with_bagging_plot')
    plt.show()
    return plot_decision_boundary, tree_clf


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Out-of-Bag evaluation
        """
    )
    return


@app.cell
def _(BaggingClassifier, DecisionTreeClassifier, X_train, y_train):
    bag_clf_1 = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500, oob_score=True, n_jobs=-1, random_state=42)
    bag_clf_1.fit(X_train, y_train)
    bag_clf_1.oob_score_
    return (bag_clf_1,)


@app.cell
def _(bag_clf_1):
    bag_clf_1.oob_decision_function_[:3]
    return


@app.cell
def _(X_test, bag_clf_1, y_test):
    from sklearn.metrics import accuracy_score
    y_pred = bag_clf_1.predict(X_test)
    accuracy_score(y_test, y_pred)
    return accuracy_score, y_pred


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If you randomly draw one instance from a dataset of size _m_, each instance in the dataset obviously has probability 1/_m_ of getting picked, and therefore it has a probability 1 – 1/_m_ of _not_ getting picked. If you draw _m_ instances with replacement, all draws are independent and therefore each instance has a probability (1 – 1/_m_)<sup>_m_</sup> of _not_ getting picked. Now let's use the fact that exp(_x_) is equal to the limit of (1 + _x_/_m_)<sup>_m_</sup> as _m_ approaches infinity. So if _m_ is large, the ratio of out-of-bag instances will be about exp(–1) ≈ 0.37. So roughly 63% (1 – 0.37) will be sampled.
        """
    )
    return


@app.cell
def _(np):
    # extra code – shows how to compute the 63% proba
    print(1 - (1 - 1 / 1000) ** 1000)
    print(1 - np.exp(-1))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Random Forests
        """
    )
    return


@app.cell
def _(RandomForestClassifier_1, X_test, X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    _rnd_clf = RandomForestClassifier_1(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
    _rnd_clf.fit(X_train, y_train)
    y_pred_rf = _rnd_clf.predict(X_test)
    return RandomForestClassifier, y_pred_rf


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A Random Forest is equivalent to a bag of decision trees:
        """
    )
    return


@app.cell
def _(BaggingClassifier, DecisionTreeClassifier):
    bag_clf_2 = BaggingClassifier(DecisionTreeClassifier(max_features='sqrt', max_leaf_nodes=16), n_estimators=500, n_jobs=-1, random_state=42)
    return (bag_clf_2,)


@app.cell
def _(X_test, X_train, bag_clf_2, np, y_pred_rf, y_train):
    bag_clf_2.fit(X_train, y_train)
    y_pred_bag = bag_clf_2.predict(X_test)
    np.all(y_pred_bag == y_pred_rf)
    return (y_pred_bag,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Feature Importance
        """
    )
    return


@app.cell
def _(RandomForestClassifier_1):
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    _rnd_clf = RandomForestClassifier_1(n_estimators=500, random_state=42)
    _rnd_clf.fit(iris.data, iris.target)
    for score, _name in zip(_rnd_clf.feature_importances_, iris.data.columns):
        print(round(score, 2), _name)
    return iris, load_iris, score


@app.cell
def _(RandomForestClassifier_1, plt, save_fig):
    from sklearn.datasets import fetch_openml
    X_mnist, y_mnist = fetch_openml('mnist_784', return_X_y=True, as_frame=False, parser='auto')
    _rnd_clf = RandomForestClassifier_1(n_estimators=100, random_state=42)
    _rnd_clf.fit(X_mnist, y_mnist)
    heatmap_image = _rnd_clf.feature_importances_.reshape(28, 28)
    plt.imshow(heatmap_image, cmap='hot')
    cbar = plt.colorbar(ticks=[_rnd_clf.feature_importances_.min(), _rnd_clf.feature_importances_.max()])
    cbar.ax.set_yticklabels(['Not important', 'Very important'], fontsize=14)
    plt.axis('off')
    save_fig('mnist_feature_importance_plot')
    plt.show()
    return X_mnist, cbar, fetch_openml, heatmap_image, y_mnist


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Boosting
        ## AdaBoost
        """
    )
    return


@app.cell
def _(SVC, X_train, np, plot_decision_boundary, plt, save_fig, y_train):
    m = len(X_train)
    _fig, _axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
    for subplot, learning_rate in ((0, 1), (1, 0.5)):
        sample_weights = np.ones(m) / m
        plt.sca(_axes[subplot])
        for i in range(5):
            svm_clf = SVC(C=0.2, gamma=0.6, random_state=42)
            svm_clf.fit(X_train, y_train, sample_weight=sample_weights * m)
            y_pred_1 = svm_clf.predict(X_train)
            error_weights = sample_weights[y_pred_1 != y_train].sum()
            r = error_weights / sample_weights.sum()
            alpha = learning_rate * np.log((1 - r) / r)
            sample_weights[y_pred_1 != y_train] = sample_weights[y_pred_1 != y_train] * np.exp(alpha)
            sample_weights = sample_weights / sample_weights.sum()
            plot_decision_boundary(svm_clf, X_train, y_train, alpha=0.4)
            plt.title(f'learning_rate = {learning_rate}')
        if subplot == 0:
            plt.text(-0.75, -0.95, '1', fontsize=16)
            plt.text(-1.05, -0.95, '2', fontsize=16)
            plt.text(1.0, -0.95, '3', fontsize=16)
            plt.text(-1.45, -0.5, '4', fontsize=16)
            plt.text(1.36, -0.95, '5', fontsize=16)
        else:
            plt.ylabel('')
    save_fig('boosting_plot')
    plt.show()
    return (
        alpha,
        error_weights,
        i,
        learning_rate,
        m,
        r,
        sample_weights,
        subplot,
        svm_clf,
        y_pred_1,
    )


@app.cell
def _(DecisionTreeClassifier, X_train, y_train):
    from sklearn.ensemble import AdaBoostClassifier

    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=30,
        learning_rate=0.5, random_state=42)
    ada_clf.fit(X_train, y_train)
    return AdaBoostClassifier, ada_clf


@app.cell
def _(X_train, ada_clf, plot_decision_boundary, y_train):
    # extra code – in case you're curious to see what the decision boundary
    #              looks like for the AdaBoost classifier
    plot_decision_boundary(ada_clf, X_train, y_train)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Gradient Boosting
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's create a simple quadratic dataset and fit a `DecisionTreeRegressor` to it:
        """
    )
    return


@app.cell
def _(np):
    from sklearn.tree import DecisionTreeRegressor
    np.random.seed(42)
    X_1 = np.random.rand(100, 1) - 0.5
    y_1 = 3 * X_1[:, 0] ** 2 + 0.05 * np.random.randn(100)
    tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg1.fit(X_1, y_1)
    return DecisionTreeRegressor, X_1, tree_reg1, y_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's train another decision tree regressor on the residual errors made by the previous predictor:
        """
    )
    return


@app.cell
def _(DecisionTreeRegressor, X_1, tree_reg1, y_1):
    y2 = y_1 - tree_reg1.predict(X_1)
    tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=43)
    tree_reg2.fit(X_1, y2)
    return tree_reg2, y2


@app.cell
def _(DecisionTreeRegressor, X_1, tree_reg2, y2):
    y3 = y2 - tree_reg2.predict(X_1)
    tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=44)
    tree_reg3.fit(X_1, y3)
    return tree_reg3, y3


@app.cell
def _(np, tree_reg1, tree_reg2, tree_reg3):
    X_new = np.array([[-0.4], [0.], [0.5]])
    sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
    return (X_new,)


@app.cell
def _(X_1, np, plt, save_fig, tree_reg1, tree_reg2, tree_reg3, y2, y3, y_1):
    def plot_predictions(regressors, X, y, axes, style, label=None, data_style='b.', data_label=None):
        x1 = np.linspace(_axes[0], _axes[1], 500)
        y_pred = sum((regressor.predict(x1.reshape(-1, 1)) for regressor in regressors))
        plt.plot(X[:, 0], y, data_style, label=data_label)
        plt.plot(x1, y_pred, style, linewidth=2, label=label)
        if label or data_label:
            plt.legend(loc='upper center')
        plt.axis(_axes)
    plt.figure(figsize=(11, 11))
    plt.subplot(3, 2, 1)
    plot_predictions([tree_reg1], X_1, y_1, axes=[-0.5, 0.5, -0.2, 0.8], style='g-', label='$h_1(x_1)$', data_label='Training set')
    plt.ylabel('$y$  ', rotation=0)
    plt.title('Residuals and tree predictions')
    plt.subplot(3, 2, 2)
    plot_predictions([tree_reg1], X_1, y_1, axes=[-0.5, 0.5, -0.2, 0.8], style='r-', label='$h(x_1) = h_1(x_1)$', data_label='Training set')
    plt.title('Ensemble predictions')
    plt.subplot(3, 2, 3)
    plot_predictions([tree_reg2], X_1, y2, axes=[-0.5, 0.5, -0.4, 0.6], style='g-', label='$h_2(x_1)$', data_style='k+', data_label='Residuals: $y - h_1(x_1)$')
    plt.ylabel('$y$  ', rotation=0)
    plt.subplot(3, 2, 4)
    plot_predictions([tree_reg1, tree_reg2], X_1, y_1, axes=[-0.5, 0.5, -0.2, 0.8], style='r-', label='$h(x_1) = h_1(x_1) + h_2(x_1)$')
    plt.subplot(3, 2, 5)
    plot_predictions([tree_reg3], X_1, y3, axes=[-0.5, 0.5, -0.4, 0.6], style='g-', label='$h_3(x_1)$', data_style='k+', data_label='Residuals: $y - h_1(x_1) - h_2(x_1)$')
    plt.xlabel('$x_1$')
    plt.ylabel('$y$  ', rotation=0)
    plt.subplot(3, 2, 6)
    plot_predictions([tree_reg1, tree_reg2, tree_reg3], X_1, y_1, axes=[-0.5, 0.5, -0.2, 0.8], style='r-', label='$h(x_1) = h_1(x_1) + h_2(x_1) + h_3(x_1)$')
    plt.xlabel('$x_1$')
    save_fig('gradient_boosting_plot')
    plt.show()
    return (plot_predictions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's try a gradient boosting regressor:
        """
    )
    return


@app.cell
def _(X_1, y_1):
    from sklearn.ensemble import GradientBoostingRegressor
    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
    gbrt.fit(X_1, y_1)
    return GradientBoostingRegressor, gbrt


@app.cell
def _(GradientBoostingRegressor, X_1, y_1):
    gbrt_best = GradientBoostingRegressor(max_depth=2, learning_rate=0.05, n_estimators=500, n_iter_no_change=10, random_state=42)
    gbrt_best.fit(X_1, y_1)
    return (gbrt_best,)


@app.cell
def _(gbrt_best):
    gbrt_best.n_estimators_
    return


@app.cell
def _(X_1, gbrt, gbrt_best, plot_predictions, plt, save_fig, y_1):
    _fig, _axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
    plt.sca(_axes[0])
    plot_predictions([gbrt], X_1, y_1, axes=[-0.5, 0.5, -0.1, 0.8], style='r-', label='Ensemble predictions')
    plt.title(f'learning_rate={gbrt.learning_rate}, n_estimators={gbrt.n_estimators_}')
    plt.xlabel('$x_1$')
    plt.ylabel('$y$', rotation=0)
    plt.sca(_axes[1])
    plot_predictions([gbrt_best], X_1, y_1, axes=[-0.5, 0.5, -0.1, 0.8], style='r-')
    plt.title(f'learning_rate={gbrt_best.learning_rate}, n_estimators={gbrt_best.n_estimators_}')
    plt.xlabel('$x_1$')
    save_fig('gbrt_learning_rate_plot')
    plt.show()
    return


@app.cell
def _(Path, train_test_split):
    # extra code – at least not in this chapter, it's presented in chapter 2

    import pandas as pd
    import tarfile
    import urllib.request

    def load_housing_data():
        tarball_path = Path("datasets/housing.tgz")
        if not tarball_path.is_file():
            Path("datasets").mkdir(parents=True, exist_ok=True)
            url = "https://github.com/ageron/data/raw/main/housing.tgz"
            urllib.request.urlretrieve(url, tarball_path)
            with tarfile.open(tarball_path) as housing_tarball:
                housing_tarball.extractall(path="datasets")
        return pd.read_csv(Path("datasets/housing/housing.csv"))

    housing = load_housing_data()

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    housing_labels = train_set["median_house_value"]
    housing = train_set.drop("median_house_value", axis=1)
    return (
        housing,
        housing_labels,
        load_housing_data,
        pd,
        tarfile,
        test_set,
        train_set,
        urllib,
    )


@app.cell
def _(housing, housing_labels):
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import make_column_transformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import OrdinalEncoder 

    hgb_reg = make_pipeline(
        make_column_transformer((OrdinalEncoder(), ["ocean_proximity"]),
                                remainder="passthrough"),
        HistGradientBoostingRegressor(categorical_features=[0], random_state=42)
    )
    hgb_reg.fit(housing, housing_labels)
    return (
        HistGradientBoostingRegressor,
        OrdinalEncoder,
        hgb_reg,
        make_column_transformer,
        make_pipeline,
    )


@app.cell
def _(hgb_reg, housing, housing_labels, pd):
    # extra code – evaluate the RMSE stats for the hgb_reg model

    from sklearn.model_selection import cross_val_score

    hgb_rmses = -cross_val_score(hgb_reg, housing, housing_labels,
                                 scoring="neg_root_mean_squared_error", cv=10)
    pd.Series(hgb_rmses).describe()
    return cross_val_score, hgb_rmses


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Stacking
        """
    )
    return


@app.cell
def _(LogisticRegression, RandomForestClassifier_1, SVC, X_train, y_train):
    from sklearn.ensemble import StackingClassifier
    stacking_clf = StackingClassifier(estimators=[('lr', LogisticRegression(random_state=42)), ('rf', RandomForestClassifier_1(random_state=42)), ('svc', SVC(probability=True, random_state=42))], final_estimator=RandomForestClassifier_1(random_state=43), cv=5)
    stacking_clf.fit(X_train, y_train)
    return StackingClassifier, stacking_clf


@app.cell
def _(X_test, stacking_clf, y_test):
    stacking_clf.score(X_test, y_test)
    return


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
        ## 1. to 7.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        1. If you have trained five different models and they all achieve 95% precision, you can try combining them into a voting ensemble, which will often give you even better results. It works better if the models are very different (e.g., an SVM classifier, a Decision Tree classifier, a Logistic Regression classifier, and so on). It is even better if they are trained on different training instances (that's the whole point of bagging and pasting ensembles), but if not this will still be effective as long as the models are very different.
        2. A hard voting classifier just counts the votes of each classifier in the ensemble and picks the class that gets the most votes. A soft voting classifier computes the average estimated class probability for each class and picks the class with the highest probability. This gives high-confidence votes more weight and often performs better, but it works only if every classifier is able to estimate class probabilities (e.g., for the SVM classifiers in Scikit-Learn you must set `probability=True`).
        3. It is quite possible to speed up training of a bagging ensemble by distributing it across multiple servers, since each predictor in the ensemble is independent of the others. The same goes for pasting ensembles and Random Forests, for the same reason. However, each predictor in a boosting ensemble is built based on the previous predictor, so training is necessarily sequential, and you will not gain anything by distributing training across multiple servers. Regarding stacking ensembles, all the predictors in a given layer are independent of each other, so they can be trained in parallel on multiple servers. However, the predictors in one layer can only be trained after the predictors in the previous layer have all been trained.
        4. With out-of-bag evaluation, each predictor in a bagging ensemble is evaluated using instances that it was not trained on (they were held out). This makes it possible to have a fairly unbiased evaluation of the ensemble without the need for an additional validation set. Thus, you have more instances available for training, and your ensemble can perform slightly better.
        5. When you are growing a tree in a Random Forest, only a random subset of the features is considered for splitting at each node. This is true as well for Extra-Trees, but they go one step further: rather than searching for the best possible thresholds, like regular Decision Trees do, they use random thresholds for each feature. This extra randomness acts like a form of regularization: if a Random Forest overfits the training data, Extra-Trees might perform better. Moreover, since Extra-Trees don't search for the best possible thresholds, they are much faster to train than Random Forests. However, they are neither faster nor slower than Random Forests when making predictions.
        6. If your AdaBoost ensemble underfits the training data, you can try increasing the number of estimators or reducing the regularization hyperparameters of the base estimator. You may also try slightly increasing the learning rate.
        7. If your Gradient Boosting ensemble overfits the training set, you should try decreasing the learning rate. You could also use early stopping to find the right number of predictors (you probably have too many).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 8. Voting Classifier
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Exercise: _Load the MNIST data and split it into a training set, a validation set, and a test set (e.g., use 50,000 instances for training, 10,000 for validation, and 10,000 for testing)._
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The MNIST dataset was loaded earlier. The dataset is already split into a training set (the first 60,000 instances) and a test set (the last 10,000 instances), and the training set is already shuffled. So all we need to do is to take the first 50,000 instances for the new training set, the next 10,000 for the validation set, and the last 10,000 for the test set:
        """
    )
    return


@app.cell
def _(X_mnist, y_mnist):
    X_train_1, y_train_1 = (X_mnist[:50000], y_mnist[:50000])
    X_valid, y_valid = (X_mnist[50000:60000], y_mnist[50000:60000])
    X_test_1, y_test_1 = (X_mnist[60000:], y_mnist[60000:])
    return X_test_1, X_train_1, X_valid, y_test_1, y_train_1, y_valid


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Exercise: _Then train various classifiers, such as a Random Forest classifier, an Extra-Trees classifier, and an SVM._
        """
    )
    return


@app.cell
def _():
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.svm import LinearSVC
    from sklearn.neural_network import MLPClassifier
    return ExtraTreesClassifier, LinearSVC, MLPClassifier


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note: The `LinearSVC` has a `dual` hyperparameter whose default value will change from `True` to `"auto"` in Scikit-Learn 1.5. To ensure this notebook continues to produce the same outputs, I'm setting it explicitly to `True`. Please see the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) for more details.
        """
    )
    return


@app.cell
def _(
    ExtraTreesClassifier,
    LinearSVC,
    MLPClassifier,
    RandomForestClassifier_1,
):
    random_forest_clf = RandomForestClassifier_1(n_estimators=100, random_state=42)
    extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
    svm_clf_1 = LinearSVC(max_iter=100, tol=20, dual=True, random_state=42)
    mlp_clf = MLPClassifier(random_state=42)
    return extra_trees_clf, mlp_clf, random_forest_clf, svm_clf_1


@app.cell
def _(
    X_train_1,
    extra_trees_clf,
    mlp_clf,
    random_forest_clf,
    svm_clf_1,
    y_train_1,
):
    estimators = [random_forest_clf, extra_trees_clf, svm_clf_1, mlp_clf]
    for _estimator in estimators:
        print('Training the', _estimator)
        _estimator.fit(X_train_1, y_train_1)
    return (estimators,)


@app.cell
def _(X_valid, estimators, y_valid):
    [_estimator.score(X_valid, y_valid) for _estimator in estimators]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The linear SVM is far outperformed by the other classifiers. However, let's keep it for now since it may improve the voting classifier's performance.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Exercise: _Next, try to combine \[the classifiers\] into an ensemble that outperforms them all on the validation set, using a soft or hard voting classifier._
        """
    )
    return


@app.cell
def _():
    from sklearn.ensemble import VotingClassifier
    return (VotingClassifier,)


@app.cell
def _(extra_trees_clf, mlp_clf, random_forest_clf, svm_clf_1):
    named_estimators = [('random_forest_clf', random_forest_clf), ('extra_trees_clf', extra_trees_clf), ('svm_clf', svm_clf_1), ('mlp_clf', mlp_clf)]
    return (named_estimators,)


@app.cell
def _(VotingClassifier_1, named_estimators):
    voting_clf_1 = VotingClassifier_1(named_estimators)
    return (voting_clf_1,)


@app.cell
def _(X_train_1, voting_clf_1, y_train_1):
    voting_clf_1.fit(X_train_1, y_train_1)
    return


@app.cell
def _(X_valid, voting_clf_1, y_valid):
    voting_clf_1.score(X_valid, y_valid)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `VotingClassifier` made a clone of each classifier, and it trained the clones using class indices as the labels, not the original class names. Therefore, to evaluate these clones we need to provide class indices as well. To convert the classes to class indices, we can use a `LabelEncoder`:
        """
    )
    return


@app.cell
def _(y_valid):
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    y_valid_encoded = encoder.fit_transform(y_valid)
    return LabelEncoder, encoder, y_valid_encoded


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        However, in the case of MNIST, it's simpler to just convert the class names to integers, since the digits match the class ids:
        """
    )
    return


@app.cell
def _(np, y_valid):
    y_valid_encoded_1 = y_valid.astype(np.int64)
    return (y_valid_encoded_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's evaluate the classifier clones:
        """
    )
    return


@app.cell
def _(X_valid, voting_clf_1, y_valid_encoded_1):
    [_estimator.score(X_valid, y_valid_encoded_1) for _estimator in voting_clf_1.estimators_]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's remove the SVM to see if performance improves. It is possible to remove an estimator by setting it to `"drop"` using `set_params()` like this:
        """
    )
    return


@app.cell
def _(voting_clf_1):
    voting_clf_1.set_params(svm_clf='drop')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This updated the list of estimators:
        """
    )
    return


@app.cell
def _(voting_clf_1):
    voting_clf_1.estimators
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        However, it did not update the list of _trained_ estimators:
        """
    )
    return


@app.cell
def _(voting_clf_1):
    voting_clf_1.estimators_
    return


@app.cell
def _(voting_clf_1):
    voting_clf_1.named_estimators_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        So we can either fit the `VotingClassifier` again, or just remove the SVM from the list of trained estimators, both in `estimators_` and `named_estimators_`:
        """
    )
    return


@app.cell
def _(voting_clf_1):
    svm_clf_trained = voting_clf_1.named_estimators_.pop('svm_clf')
    voting_clf_1.estimators_.remove(svm_clf_trained)
    return (svm_clf_trained,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's evaluate the `VotingClassifier` again:
        """
    )
    return


@app.cell
def _(X_valid, voting_clf_1, y_valid):
    voting_clf_1.score(X_valid, y_valid)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A bit better! The SVM was hurting performance. Now let's try using a soft voting classifier. We do not actually need to retrain the classifier, we can just set `voting` to `"soft"`:
        """
    )
    return


@app.cell
def _(voting_clf_1):
    voting_clf_1.voting = 'soft'
    return


@app.cell
def _(X_valid, voting_clf_1, y_valid):
    voting_clf_1.score(X_valid, y_valid)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Nope, hard voting wins in this case.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        _Once you have found \[an ensemble that performs better than the individual predictors\], try it on the test set. How much better does it perform compared to the individual classifiers?_
        """
    )
    return


@app.cell
def _(X_test_1, voting_clf_1, y_test_1):
    voting_clf_1.voting = 'hard'
    voting_clf_1.score(X_test_1, y_test_1)
    return


@app.cell
def _(X_test_1, np, voting_clf_1, y_test_1):
    [_estimator.score(X_test_1, y_test_1.astype(np.int64)) for _estimator in voting_clf_1.estimators_]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The voting classifier reduced the error rate of the best model from about 3% to 2.7%, which means 10% less errors.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 9. Stacking Ensemble
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Exercise: _Run the individual classifiers from the previous exercise to make predictions on the validation set, and create a new training set with the resulting predictions: each training instance is a vector containing the set of predictions from all your classifiers for an image, and the target is the image's class. Train a classifier on this new training set._
        """
    )
    return


@app.cell
def _(X_valid, estimators, np):
    X_valid_predictions = np.empty((len(X_valid), len(estimators)), dtype=object)
    for _index, _estimator in enumerate(estimators):
        X_valid_predictions[:, _index] = _estimator.predict(X_valid)
    return (X_valid_predictions,)


@app.cell
def _(X_valid_predictions):
    X_valid_predictions
    return


@app.cell
def _(RandomForestClassifier_1, X_valid_predictions, y_valid):
    rnd_forest_blender = RandomForestClassifier_1(n_estimators=200, oob_score=True, random_state=42)
    rnd_forest_blender.fit(X_valid_predictions, y_valid)
    return (rnd_forest_blender,)


@app.cell
def _(rnd_forest_blender):
    rnd_forest_blender.oob_score_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        You could fine-tune this blender or try other types of blenders (e.g., an `MLPClassifier`), then select the best one using cross-validation, as always.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Exercise: _Congratulations, you have just trained a blender, and together with the classifiers they form a stacking ensemble! Now let's evaluate the ensemble on the test set. For each image in the test set, make predictions with all your classifiers, then feed the predictions to the blender to get the ensemble's predictions. How does it compare to the voting classifier you trained earlier?_
        """
    )
    return


@app.cell
def _(X_test_1, estimators, np):
    X_test_predictions = np.empty((len(X_test_1), len(estimators)), dtype=object)
    for _index, _estimator in enumerate(estimators):
        X_test_predictions[:, _index] = _estimator.predict(X_test_1)
    return (X_test_predictions,)


@app.cell
def _(X_test_predictions, rnd_forest_blender):
    y_pred_2 = rnd_forest_blender.predict(X_test_predictions)
    return (y_pred_2,)


@app.cell
def _(accuracy_score, y_pred_2, y_test_1):
    accuracy_score(y_test_1, y_pred_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This stacking ensemble does not perform as well as the voting classifier we trained earlier.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Exercise: _Now try again using a `StackingClassifier` instead: do you get better performance? If so, why?_
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Since `StackingClassifier` uses K-Fold cross-validation, we don't need a separate validation set, so let's join the training set and the validation set into a bigger training set:
        """
    )
    return


@app.cell
def _(X_mnist, y_mnist):
    X_train_full, y_train_full = X_mnist[:60_000], y_mnist[:60_000]
    return X_train_full, y_train_full


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's create and train the stacking classifier on the full training set:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Warning**: the following cell will take quite a while to run (15-30 minutes depending on your hardware), as it uses K-Fold validation with 5 folds by default. It will train the 4 classifiers 5 times each on 80% of the full training set to make the predictions, plus one last time each on the full training set, and lastly it will train the final model on the predictions. That's a total of 25 models to train!
        """
    )
    return


@app.cell
def _(
    StackingClassifier,
    X_train_full,
    named_estimators,
    rnd_forest_blender,
    y_train_full,
):
    stack_clf = StackingClassifier(named_estimators,
                                   final_estimator=rnd_forest_blender)
    stack_clf.fit(X_train_full, y_train_full)
    return (stack_clf,)


@app.cell
def _(X_test_1, stack_clf, y_test_1):
    stack_clf.score(X_test_1, y_test_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `StackingClassifier` significantly outperforms the custom stacking implementation we tried earlier! This is for mainly two reasons:

        * Since we could reclaim the validation set, the `StackingClassifier` was trained on a larger dataset.
        * It used `predict_proba()` if available, or else `decision_function()` if available, or else `predict()`. This gave the blender much more nuanced inputs to work with.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And that's all for today, congratulations on finishing the chapter and the exercises!
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

