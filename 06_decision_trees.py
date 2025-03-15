import marimo

__generated_with = "0.11.20"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Chapter 6 – Decision Trees**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        _This notebook contains all the sample code and solutions to the exercises in chapter 6._
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <table align="left">
          <td>
            <a href="https://colab.research.google.com/github/ageron/handson-ml3/blob/main/06_decision_trees.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
          </td>
          <td>
            <a target="_blank" href="https://kaggle.com/kernels/welcome?src=https://github.com/ageron/handson-ml3/blob/main/06_decision_trees.ipynb"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>
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
        And let's create the `images/decision_trees` folder (if it doesn't already exist), and define the `save_fig()` function which is used through this notebook to save the figures in high-res for the book:
        """
    )
    return


@app.cell
def _(plt):
    from pathlib import Path

    IMAGES_PATH = Path() / "images" / "decision_trees"
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
        # Training and Visualizing a Decision Tree
        """
    )
    return


@app.cell
def _():
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier

    iris = load_iris(as_frame=True)
    X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
    y_iris = iris.target

    tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree_clf.fit(X_iris, y_iris)
    return DecisionTreeClassifier, X_iris, iris, load_iris, tree_clf, y_iris


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **This code example generates Figure 6–1. Iris Decision Tree:**
        """
    )
    return


@app.cell
def _(IMAGES_PATH, iris, tree_clf):
    from sklearn.tree import export_graphviz

    export_graphviz(
            tree_clf,
            out_file=str(IMAGES_PATH / "iris_tree.dot"),  # path differs in the book
            feature_names=["petal length (cm)", "petal width (cm)"],
            class_names=iris.target_names,
            rounded=True,
            filled=True
        )
    return (export_graphviz,)


@app.cell
def _(IMAGES_PATH):
    from graphviz import Source

    Source.from_file(IMAGES_PATH / "iris_tree.dot")  # path differs in the book
    return (Source,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Graphviz also provides the `dot` command line tool to convert `.dot` files to a variety of formats. The following command converts the dot file to a png image:
        """
    )
    return


app._unparsable_cell(
    r"""
    # extra code
    !dot -Tpng {IMAGES_PATH / \"iris_tree.dot\"} -o {IMAGES_PATH / \"iris_tree.png\"}
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Making Predictions
        """
    )
    return


@app.cell
def _(DecisionTreeClassifier, X_iris, iris, plt, save_fig, tree_clf, y_iris):
    import numpy as np

    # extra code – just formatting details
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.figure(figsize=(8, 4))

    lengths, widths = np.meshgrid(np.linspace(0, 7.2, 100), np.linspace(0, 3, 100))
    X_iris_all = np.c_[lengths.ravel(), widths.ravel()]
    y_pred = tree_clf.predict(X_iris_all).reshape(lengths.shape)
    plt.contourf(lengths, widths, y_pred, alpha=0.3, cmap=custom_cmap)
    for idx, (name, style) in enumerate(zip(iris.target_names, ("yo", "bs", "g^"))):
        plt.plot(X_iris[:, 0][y_iris == idx], X_iris[:, 1][y_iris == idx],
                 style, label=f"Iris {name}")

    # extra code – this section beautifies and saves Figure 6–2
    tree_clf_deeper = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_clf_deeper.fit(X_iris, y_iris)
    th0, th1, th2a, th2b = tree_clf_deeper.tree_.threshold[[0, 2, 3, 6]]
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Petal width (cm)")
    plt.plot([th0, th0], [0, 3], "k-", linewidth=2)
    plt.plot([th0, 7.2], [th1, th1], "k--", linewidth=2)
    plt.plot([th2a, th2a], [0, th1], "k:", linewidth=2)
    plt.plot([th2b, th2b], [th1, 3], "k:", linewidth=2)
    plt.text(th0 - 0.05, 1.0, "Depth=0", horizontalalignment="right", fontsize=15)
    plt.text(3.2, th1 + 0.02, "Depth=1", verticalalignment="bottom", fontsize=13)
    plt.text(th2a + 0.05, 0.5, "(Depth=2)", fontsize=11)
    plt.axis([0, 7.2, 0, 3])
    plt.legend()
    save_fig("decision_tree_decision_boundaries_plot")

    plt.show()
    return (
        ListedColormap,
        X_iris_all,
        custom_cmap,
        idx,
        lengths,
        name,
        np,
        style,
        th0,
        th1,
        th2a,
        th2b,
        tree_clf_deeper,
        widths,
        y_pred,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        You can access the tree structure via the `tree_` attribute:
        """
    )
    return


@app.cell
def _(tree_clf):
    tree_clf.tree_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        For more information, check out this class's documentation:
        """
    )
    return


@app.cell
def _():
    # help(sklearn.tree._tree.Tree)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        See the extra material section below for an example.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Estimating Class Probabilities
        """
    )
    return


@app.cell
def _(tree_clf):
    tree_clf.predict_proba([[5, 1.5]]).round(3)
    return


@app.cell
def _(tree_clf):
    tree_clf.predict([[5, 1.5]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Regularization Hyperparameters
        """
    )
    return


@app.cell
def _(DecisionTreeClassifier):
    from sklearn.datasets import make_moons

    X_moons, y_moons = make_moons(n_samples=150, noise=0.2, random_state=42)

    tree_clf1 = DecisionTreeClassifier(random_state=42)
    tree_clf2 = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)
    tree_clf1.fit(X_moons, y_moons)
    tree_clf2.fit(X_moons, y_moons)
    return X_moons, make_moons, tree_clf1, tree_clf2, y_moons


@app.cell
def _(X_moons, np, plt, save_fig, tree_clf1, tree_clf2, y_moons):
    # extra code – this cell generates and saves Figure 6–3

    def plot_decision_boundary(clf, X, y, axes, cmap):
        x1, x2 = np.meshgrid(np.linspace(axes[0], axes[1], 100),
                             np.linspace(axes[2], axes[3], 100))
        X_new = np.c_[x1.ravel(), x2.ravel()]
        y_pred = clf.predict(X_new).reshape(x1.shape)
    
        plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=cmap)
        plt.contour(x1, x2, y_pred, cmap="Greys", alpha=0.8)
        colors = {"Wistia": ["#78785c", "#c47b27"], "Pastel1": ["red", "blue"]}
        markers = ("o", "^")
        for idx in (0, 1):
            plt.plot(X[:, 0][y == idx], X[:, 1][y == idx],
                     color=colors[cmap][idx], marker=markers[idx], linestyle="none")
        plt.axis(axes)
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$", rotation=0)

    fig, axes = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
    plt.sca(axes[0])
    plot_decision_boundary(tree_clf1, X_moons, y_moons,
                           axes=[-1.5, 2.4, -1, 1.5], cmap="Wistia")
    plt.title("No restrictions")
    plt.sca(axes[1])
    plot_decision_boundary(tree_clf2, X_moons, y_moons,
                           axes=[-1.5, 2.4, -1, 1.5], cmap="Wistia")
    plt.title(f"min_samples_leaf = {tree_clf2.min_samples_leaf}")
    plt.ylabel("")
    save_fig("min_samples_leaf_plot")
    plt.show()
    return axes, fig, plot_decision_boundary


@app.cell
def _(make_moons, tree_clf1):
    X_moons_test, y_moons_test = make_moons(n_samples=1000, noise=0.2,
                                            random_state=43)
    tree_clf1.score(X_moons_test, y_moons_test)
    return X_moons_test, y_moons_test


@app.cell
def _(X_moons_test, tree_clf2, y_moons_test):
    tree_clf2.score(X_moons_test, y_moons_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Regression
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's prepare a simple quadratic training set:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Code example:**
        """
    )
    return


@app.cell
def _(np):
    from sklearn.tree import DecisionTreeRegressor

    np.random.seed(42)
    X_quad = np.random.rand(200, 1) - 0.5  # a single random input feature
    y_quad = X_quad ** 2 + 0.025 * np.random.randn(200, 1)

    tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
    tree_reg.fit(X_quad, y_quad)
    return DecisionTreeRegressor, X_quad, tree_reg, y_quad


@app.cell
def _(IMAGES_PATH, Source, export_graphviz, tree_reg):
    # extra code – we've already seen how to use export_graphviz()
    export_graphviz(
        tree_reg,
        out_file=str(IMAGES_PATH / "regression_tree.dot"),
        feature_names=["x1"],
        rounded=True,
        filled=True
    )
    Source.from_file(IMAGES_PATH / "regression_tree.dot")
    return


@app.cell
def _(DecisionTreeRegressor, X_quad, y_quad):
    tree_reg2 = DecisionTreeRegressor(max_depth=3, random_state=42)
    tree_reg2.fit(X_quad, y_quad)
    return (tree_reg2,)


@app.cell
def _(tree_reg):
    tree_reg.tree_.threshold
    return


@app.cell
def _(tree_reg2):
    tree_reg2.tree_.threshold
    return


@app.cell
def _(X_quad, np, plt, save_fig, tree_reg, tree_reg2, y_quad):
    def plot_regression_predictions(tree_reg, X, y, axes=[-0.5, 0.5, -0.05, 0.25]):
        x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
        y_pred = tree_reg.predict(x1)
        plt.axis(axes)
        plt.xlabel('$x_1$')
        plt.plot(X, y, 'b.')
        plt.plot(x1, y_pred, 'r.-', linewidth=2, label='$\\hat{y}$')
    fig_1, axes_1 = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
    plt.sca(axes_1[0])
    plot_regression_predictions(tree_reg, X_quad, y_quad)
    th0_1, th1a, th1b = tree_reg.tree_.threshold[[0, 1, 4]]
    for split, style_1 in ((th0_1, 'k-'), (th1a, 'k--'), (th1b, 'k--')):
        plt.plot([split, split], [-0.05, 0.25], style_1, linewidth=2)
    plt.text(th0_1, 0.16, 'Depth=0', fontsize=15)
    plt.text(th1a + 0.01, -0.01, 'Depth=1', horizontalalignment='center', fontsize=13)
    plt.text(th1b + 0.01, -0.01, 'Depth=1', fontsize=13)
    plt.ylabel('$y$', rotation=0)
    plt.legend(loc='upper center', fontsize=16)
    plt.title('max_depth=2')
    plt.sca(axes_1[1])
    th2s = tree_reg2.tree_.threshold[[2, 5, 9, 12]]
    plot_regression_predictions(tree_reg2, X_quad, y_quad)
    for split, style_1 in ((th0_1, 'k-'), (th1a, 'k--'), (th1b, 'k--')):
        plt.plot([split, split], [-0.05, 0.25], style_1, linewidth=2)
    for split in th2s:
        plt.plot([split, split], [-0.05, 0.25], 'k:', linewidth=1)
    plt.text(th2s[2] + 0.01, 0.15, 'Depth=2', fontsize=13)
    plt.title('max_depth=3')
    save_fig('tree_regression_plot')
    plt.show()
    return (
        axes_1,
        fig_1,
        plot_regression_predictions,
        split,
        style_1,
        th0_1,
        th1a,
        th1b,
        th2s,
    )


@app.cell
def _(DecisionTreeRegressor, X_quad, np, plt, save_fig, y_quad):
    tree_reg1 = DecisionTreeRegressor(random_state=42)
    tree_reg2_1 = DecisionTreeRegressor(random_state=42, min_samples_leaf=10)
    tree_reg1.fit(X_quad, y_quad)
    tree_reg2_1.fit(X_quad, y_quad)
    x1 = np.linspace(-0.5, 0.5, 500).reshape(-1, 1)
    y_pred1 = tree_reg1.predict(x1)
    y_pred2 = tree_reg2_1.predict(x1)
    fig_2, axes_2 = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
    plt.sca(axes_2[0])
    plt.plot(X_quad, y_quad, 'b.')
    plt.plot(x1, y_pred1, 'r.-', linewidth=2, label='$\\hat{y}$')
    plt.axis([-0.5, 0.5, -0.05, 0.25])
    plt.xlabel('$x_1$')
    plt.ylabel('$y$', rotation=0)
    plt.legend(loc='upper center')
    plt.title('No restrictions')
    plt.sca(axes_2[1])
    plt.plot(X_quad, y_quad, 'b.')
    plt.plot(x1, y_pred2, 'r.-', linewidth=2, label='$\\hat{y}$')
    plt.axis([-0.5, 0.5, -0.05, 0.25])
    plt.xlabel('$x_1$')
    plt.title(f'min_samples_leaf={tree_reg2_1.min_samples_leaf}')
    save_fig('tree_regression_regularization_plot')
    plt.show()
    return axes_2, fig_2, tree_reg1, tree_reg2_1, x1, y_pred1, y_pred2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Sensitivity to axis orientation
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Rotating the dataset also leads to completely different decision boundaries:
        """
    )
    return


@app.cell
def _(DecisionTreeClassifier, np, plot_decision_boundary, plt, save_fig):
    np.random.seed(6)
    X_square = np.random.rand(100, 2) - 0.5
    y_square = (X_square[:, 0] > 0).astype(np.int64)
    angle = np.pi / 4
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    X_rotated_square = X_square.dot(rotation_matrix)
    tree_clf_square = DecisionTreeClassifier(random_state=42)
    tree_clf_square.fit(X_square, y_square)
    tree_clf_rotated_square = DecisionTreeClassifier(random_state=42)
    tree_clf_rotated_square.fit(X_rotated_square, y_square)
    fig_3, axes_3 = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)
    plt.sca(axes_3[0])
    plot_decision_boundary(tree_clf_square, X_square, y_square, axes=[-0.7, 0.7, -0.7, 0.7], cmap='Pastel1')
    plt.sca(axes_3[1])
    plot_decision_boundary(tree_clf_rotated_square, X_rotated_square, y_square, axes=[-0.7, 0.7, -0.7, 0.7], cmap='Pastel1')
    plt.ylabel('')
    save_fig('sensitivity_to_rotation_plot')
    plt.show()
    return (
        X_rotated_square,
        X_square,
        angle,
        axes_3,
        fig_3,
        rotation_matrix,
        tree_clf_rotated_square,
        tree_clf_square,
        y_square,
    )


@app.cell
def _(DecisionTreeClassifier, X_iris, y_iris):
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pca_pipeline = make_pipeline(StandardScaler(), PCA())
    X_iris_rotated = pca_pipeline.fit_transform(X_iris)
    tree_clf_pca = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree_clf_pca.fit(X_iris_rotated, y_iris)
    return (
        PCA,
        StandardScaler,
        X_iris_rotated,
        make_pipeline,
        pca_pipeline,
        tree_clf_pca,
    )


@app.cell
def _(
    X_iris_rotated,
    custom_cmap,
    iris,
    np,
    plt,
    save_fig,
    tree_clf_pca,
    y_iris,
):
    plt.figure(figsize=(8, 4))
    axes_4 = [-2.2, 2.4, -0.6, 0.7]
    z0s, z1s = np.meshgrid(np.linspace(axes_4[0], axes_4[1], 100), np.linspace(axes_4[2], axes_4[3], 100))
    X_iris_pca_all = np.c_[z0s.ravel(), z1s.ravel()]
    y_pred_1 = tree_clf_pca.predict(X_iris_pca_all).reshape(z0s.shape)
    plt.contourf(z0s, z1s, y_pred_1, alpha=0.3, cmap=custom_cmap)
    for idx_1, (name_1, style_2) in enumerate(zip(iris.target_names, ('yo', 'bs', 'g^'))):
        plt.plot(X_iris_rotated[:, 0][y_iris == idx_1], X_iris_rotated[:, 1][y_iris == idx_1], style_2, label=f'Iris {name_1}')
    plt.xlabel('$z_1$')
    plt.ylabel('$z_2$', rotation=0)
    th1_1, th2 = tree_clf_pca.tree_.threshold[[0, 2]]
    plt.plot([th1_1, th1_1], axes_4[2:], 'k-', linewidth=2)
    plt.plot([th2, th2], axes_4[2:], 'k--', linewidth=2)
    plt.text(th1_1 - 0.01, axes_4[2] + 0.05, 'Depth=0', horizontalalignment='right', fontsize=15)
    plt.text(th2 - 0.01, axes_4[2] + 0.05, 'Depth=1', horizontalalignment='right', fontsize=13)
    plt.axis(axes_4)
    plt.legend(loc=(0.32, 0.67))
    save_fig('pca_preprocessing_plot')
    plt.show()
    return (
        X_iris_pca_all,
        axes_4,
        idx_1,
        name_1,
        style_2,
        th1_1,
        th2,
        y_pred_1,
        z0s,
        z1s,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Decision Trees Have High Variance
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We've seen that small changes in the dataset (such as a rotation) may produce a very different Decision Tree.
        Now let's show that training the same model on the same data may produce a very different model every time, since the CART training algorithm used by Scikit-Learn is stochastic. To show this, we will set `random_state` to a different value than earlier:
        """
    )
    return


@app.cell
def _(DecisionTreeClassifier, X_iris, y_iris):
    tree_clf_tweaked = DecisionTreeClassifier(max_depth=2, random_state=40)
    tree_clf_tweaked.fit(X_iris, y_iris)
    return (tree_clf_tweaked,)


@app.cell
def _(
    X_iris,
    X_iris_all,
    custom_cmap,
    iris,
    lengths,
    plt,
    save_fig,
    tree_clf_tweaked,
    widths,
    y_iris,
):
    plt.figure(figsize=(8, 4))
    y_pred_2 = tree_clf_tweaked.predict(X_iris_all).reshape(lengths.shape)
    plt.contourf(lengths, widths, y_pred_2, alpha=0.3, cmap=custom_cmap)
    for idx_2, (name_2, style_3) in enumerate(zip(iris.target_names, ('yo', 'bs', 'g^'))):
        plt.plot(X_iris[:, 0][y_iris == idx_2], X_iris[:, 1][y_iris == idx_2], style_3, label=f'Iris {name_2}')
    th0_2, th1_2 = tree_clf_tweaked.tree_.threshold[[0, 2]]
    plt.plot([0, 7.2], [th0_2, th0_2], 'k-', linewidth=2)
    plt.plot([0, 7.2], [th1_2, th1_2], 'k--', linewidth=2)
    plt.text(1.8, th0_2 + 0.05, 'Depth=0', verticalalignment='bottom', fontsize=15)
    plt.text(2.3, th1_2 + 0.05, 'Depth=1', verticalalignment='bottom', fontsize=13)
    plt.xlabel('Petal length (cm)')
    plt.ylabel('Petal width (cm)')
    plt.axis([0, 7.2, 0, 3])
    plt.legend()
    save_fig('decision_tree_high_variance_plot')
    plt.show()
    return idx_2, name_2, style_3, th0_2, th1_2, y_pred_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Extra Material – Accessing the tree structure
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A trained `DecisionTreeClassifier` has a `tree_` attribute that stores the tree's structure:
        """
    )
    return


@app.cell
def _(tree_clf):
    tree = tree_clf.tree_
    tree
    return (tree,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        You can get the total number of nodes in the tree:
        """
    )
    return


@app.cell
def _(tree):
    tree.node_count
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And other self-explanatory attributes are available:
        """
    )
    return


@app.cell
def _(tree):
    tree.max_depth
    return


@app.cell
def _(tree):
    tree.max_n_classes
    return


@app.cell
def _(tree):
    tree.n_features
    return


@app.cell
def _(tree):
    tree.n_outputs
    return


@app.cell
def _(tree):
    tree.n_leaves
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        All the information about the nodes is stored in NumPy arrays. For example, the impurity of each node:
        """
    )
    return


@app.cell
def _(tree):
    tree.impurity
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The root node is at index 0. The left and right children nodes of node _i_ are `tree.children_left[i]` and `tree.children_right[i]`. For example, the children of the root node are:
        """
    )
    return


@app.cell
def _(tree):
    tree.children_left[0], tree.children_right[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        When the left and right nodes are equal, it means this is a leaf node (and the children node ids are arbitrary):
        """
    )
    return


@app.cell
def _(tree):
    tree.children_left[3], tree.children_right[3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        So you can get the leaf node ids like this:
        """
    )
    return


@app.cell
def _(np, tree):
    is_leaf = (tree.children_left == tree.children_right)
    np.arange(tree.node_count)[is_leaf]
    return (is_leaf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Non-leaf nodes are called _split nodes_. The feature they split is available via the `feature` array. Values for leaf nodes should be ignored:
        """
    )
    return


@app.cell
def _(tree):
    tree.feature
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And the corresponding thresholds are:
        """
    )
    return


@app.cell
def _(tree):
    tree.threshold
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And the number of instances per class that reached each node is available too:
        """
    )
    return


@app.cell
def _(tree):
    tree.value
    return


@app.cell
def _(tree):
    tree.n_node_samples
    return


@app.cell
def _(np, tree):
    np.all(tree.value.sum(axis=(1, 2)) == tree.n_node_samples)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here's how you can compute the depth of each node:
        """
    )
    return


@app.cell
def _(np, tree_clf):
    def compute_depth(tree_clf):
        tree = tree_clf.tree_
        depth = np.zeros(tree.node_count)
        stack = [(0, 0)]
        while stack:
            node, node_depth = stack.pop()
            depth[node] = node_depth
            if tree.children_left[node] != tree.children_right[node]:
                stack.append((tree.children_left[node], node_depth + 1))
                stack.append((tree.children_right[node], node_depth + 1))
        return depth

    depth = compute_depth(tree_clf)
    depth
    return compute_depth, depth


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here's how to get the thresholds of all split nodes at depth 1:
        """
    )
    return


@app.cell
def _(depth, is_leaf, tree_clf):
    tree_clf.tree_.feature[(depth == 1) & (~is_leaf)]
    return


@app.cell
def _(depth, is_leaf, tree_clf):
    tree_clf.tree_.threshold[(depth == 1) & (~is_leaf)]
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
        ## 1. to 6.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        1. The depth of a well-balanced binary tree containing _m_ leaves is equal to log₂(_m_), rounded up. log₂ is the binary log; log₂(_m_) = log(_m_) / log(2). A binary Decision Tree (one that makes only binary decisions, as is the case with all trees in Scikit-Learn) will end up more or less well balanced at the end of training, with one leaf per training instance if it is trained without restrictions. Thus, if the training set contains one million instances, the Decision Tree will have a depth of log₂(10<sup>6</sup>) ≈ 20 (actually a bit more since the tree will generally not be perfectly well balanced).
        2. A node's Gini impurity is generally lower than its parent's. This is due to the CART training algorithm's cost function, which splits each node in a way that minimizes the weighted sum of its children's Gini impurities. However, it is possible for a node to have a higher Gini impurity than its parent, as long as this increase is more than compensated for by a decrease in the other child's impurity. For example, consider a node containing four instances of class A and one of class B. Its Gini impurity is 1 – (1/5)² – (4/5)² = 0.32. Now suppose the dataset is one-dimensional and the instances are lined up in the following order: A, B, A, A, A. You can verify that the algorithm will split this node after the second instance, producing one child node with instances A, B, and the other child node with instances A, A, A. The first child node's Gini impurity is 1 – (1/2)² – (1/2)² = 0.5, which is higher than its parent's. This is compensated for by the fact that the other node is pure, so its overall weighted Gini impurity is 2/5 × 0.5 + 3/5 × 0 = 0.2, which is lower than the parent's Gini impurity.
        3. If a Decision Tree is overfitting the training set, it may be a good idea to decrease `max_depth`, since this will constrain the model, regularizing it.
        4. Decision Trees don't care whether or not the training data is scaled or centered; that's one of the nice things about them. So if a Decision Tree underfits the training set, scaling the input features will just be a waste of time.
        5. The computational complexity of training a Decision Tree is _O_(_n_ × _m_ log₂(_m_)). So if you multiply the training set size by 10, the training time will be multiplied by _K_ = (_n_ × 10 _m_ × log₂(10 _m_)) / (_n_ × _m_ × log₂(_m_)) = 10 × log₂(10 _m_) / log₂(_m_). If _m_ = 10<sup>6</sup>, then _K_ ≈ 11.7, so you can expect the training time to be roughly 11.7 hours.
        6. If the number of features doubles, then the training time will also roughly double.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 7.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        _Exercise: train and fine-tune a Decision Tree for the moons dataset._
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        a. Generate a moons dataset using `make_moons(n_samples=10000, noise=0.4)`.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Adding `random_state=42` to make this notebook's output constant:
        """
    )
    return


@app.cell
def _(make_moons):
    X_moons_1, y_moons_1 = make_moons(n_samples=10000, noise=0.4, random_state=42)
    return X_moons_1, y_moons_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        b. Split it into a training set and a test set using `train_test_split()`.
        """
    )
    return


@app.cell
def _(X_moons_1, y_moons_1):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_moons_1, y_moons_1, test_size=0.2, random_state=42)
    return X_test, X_train, train_test_split, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        c. Use grid search with cross-validation (with the help of the `GridSearchCV` class) to find good hyperparameter values for a `DecisionTreeClassifier`. Hint: try various values for `max_leaf_nodes`.
        """
    )
    return


@app.cell
def _(DecisionTreeClassifier, X_train, y_train):
    from sklearn.model_selection import GridSearchCV

    params = {
        'max_leaf_nodes': list(range(2, 100)),
        'max_depth': list(range(1, 7)),
        'min_samples_split': [2, 3, 4]
    }
    grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42),
                                  params,
                                  cv=3)

    grid_search_cv.fit(X_train, y_train)
    return GridSearchCV, grid_search_cv, params


@app.cell
def _(grid_search_cv):
    grid_search_cv.best_estimator_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        d. Train it on the full training set using these hyperparameters, and measure your model's performance on the test set. You should get roughly 85% to 87% accuracy.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        By default, `GridSearchCV` trains the best model found on the whole training set (you can change this by setting `refit=False`), so we don't need to do it again. We can simply evaluate the model's accuracy:
        """
    )
    return


@app.cell
def _(X_test, grid_search_cv, y_test):
    from sklearn.metrics import accuracy_score
    y_pred_3 = grid_search_cv.predict(X_test)
    accuracy_score(y_test, y_pred_3)
    return accuracy_score, y_pred_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 8.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        _Exercise: Grow a forest._
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        a. Continuing the previous exercise, generate 1,000 subsets of the training set, each containing 100 instances selected randomly. Hint: you can use Scikit-Learn's `ShuffleSplit` class for this.
        """
    )
    return


@app.cell
def _(X_train, y_train):
    from sklearn.model_selection import ShuffleSplit

    n_trees = 1000
    n_instances = 100

    mini_sets = []

    rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train) - n_instances,
                      random_state=42)

    for mini_train_index, mini_test_index in rs.split(X_train):
        X_mini_train = X_train[mini_train_index]
        y_mini_train = y_train[mini_train_index]
        mini_sets.append((X_mini_train, y_mini_train))
    return (
        ShuffleSplit,
        X_mini_train,
        mini_sets,
        mini_test_index,
        mini_train_index,
        n_instances,
        n_trees,
        rs,
        y_mini_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        b. Train one Decision Tree on each subset, using the best hyperparameter values found above. Evaluate these 1,000 Decision Trees on the test set. Since they were trained on smaller sets, these Decision Trees will likely perform worse than the first Decision Tree, achieving only about 80% accuracy.
        """
    )
    return


@app.cell
def _(X_test, accuracy_score, grid_search_cv, mini_sets, n_trees, np, y_test):
    from sklearn.base import clone
    forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]
    accuracy_scores = []
    for tree_1, (X_mini_train_1, y_mini_train_1) in zip(forest, mini_sets):
        tree_1.fit(X_mini_train_1, y_mini_train_1)
        y_pred_4 = tree_1.predict(X_test)
        accuracy_scores.append(accuracy_score(y_test, y_pred_4))
    np.mean(accuracy_scores)
    return (
        X_mini_train_1,
        accuracy_scores,
        clone,
        forest,
        tree_1,
        y_mini_train_1,
        y_pred_4,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        c. Now comes the magic. For each test set instance, generate the predictions of the 1,000 Decision Trees, and keep only the most frequent prediction (you can use SciPy's `mode()` function for this). This gives you _majority-vote predictions_ over the test set.
        """
    )
    return


@app.cell
def _(X_test, forest, n_trees, np):
    Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)
    for tree_index, tree_2 in enumerate(forest):
        Y_pred[tree_index] = tree_2.predict(X_test)
    return Y_pred, tree_2, tree_index


@app.cell
def _(Y_pred):
    from scipy.stats import mode

    y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)
    return mode, n_votes, y_pred_majority_votes


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        d. Evaluate these predictions on the test set: you should obtain a slightly higher accuracy than your first model (about 0.5 to 1.5% higher). Congratulations, you have trained a Random Forest classifier!
        """
    )
    return


@app.cell
def _(accuracy_score, y_pred_majority_votes, y_test):
    accuracy_score(y_test, y_pred_majority_votes.reshape([-1]))
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

