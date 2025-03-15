import marimo

__generated_with = "0.11.20"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Support Vector Machines**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        _This notebook contains all the sample code and solutions to the exercises in chapter 5._
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <table align="left">
          <td>
            <a href="https://colab.research.google.com/github/ageron/handson-ml3/blob/main/05_support_vector_machines.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
          </td>
          <td>
            <a target="_blank" href="https://kaggle.com/kernels/welcome?src=https://github.com/ageron/handson-ml3/blob/main/05_support_vector_machines.ipynb"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>
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
        And let's create the `images/svm` folder (if it doesn't already exist), and define the `save_fig()` function which is used through this notebook to save the figures in high-res for the book:
        """
    )
    return


@app.cell
def _(plt):
    from pathlib import Path

    IMAGES_PATH = Path() / "images" / "svm"
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
        # Linear SVM Classification
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The book starts with a few figures, before the first code example, so the next three cells generate and save these figures. You can skip them if you want.
        """
    )
    return


@app.cell
def _(plt, save_fig):
    import numpy as np
    from sklearn.svm import SVC
    from sklearn import datasets
    iris = datasets.load_iris(as_frame=True)
    X = iris.data[['petal length (cm)', 'petal width (cm)']].values
    y = iris.target
    _setosa_or_versicolor = (y == 0) | (y == 1)
    X = X[_setosa_or_versicolor]
    y = y[_setosa_or_versicolor]
    svm_clf = SVC(kernel='linear', C=1e+100)
    svm_clf.fit(X, y)
    x0 = np.linspace(0, 5.5, 200)
    pred_1 = 5 * x0 - 20
    pred_2 = x0 - 1.8
    pred_3 = 0.1 * x0 + 0.5

    def plot_svc_decision_boundary(svm_clf, xmin, xmax):
        w = svm_clf.coef_[0]
        b = svm_clf.intercept_[0]
        x0 = np.linspace(xmin, xmax, 200)
        decision_boundary = -w[0] / w[1] * x0 - b / w[1]
        margin = 1 / w[1]
        gutter_up = decision_boundary + margin
        gutter_down = decision_boundary - margin
        svs = svm_clf.support_vectors_
        plt.plot(x0, decision_boundary, 'k-', linewidth=2, zorder=-2)
        plt.plot(x0, gutter_up, 'k--', linewidth=2, zorder=-2)
        plt.plot(x0, gutter_down, 'k--', linewidth=2, zorder=-2)
        plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#AAA', zorder=-1)
    _fig, _axes = plt.subplots(ncols=2, figsize=(10, 2.7), sharey=True)
    plt.sca(_axes[0])
    plt.plot(x0, pred_1, 'g--', linewidth=2)
    plt.plot(x0, pred_2, 'm-', linewidth=2)
    plt.plot(x0, pred_3, 'r-', linewidth=2)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs', label='Iris versicolor')
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'yo', label='Iris setosa')
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.legend(loc='upper left')
    plt.axis([0, 5.5, 0, 2])
    plt.gca().set_aspect('equal')
    plt.grid()
    plt.sca(_axes[1])
    plot_svc_decision_boundary(svm_clf, 0, 5.5)
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'yo')
    plt.xlabel('Petal length')
    plt.axis([0, 5.5, 0, 2])
    plt.gca().set_aspect('equal')
    plt.grid()
    save_fig('large_margin_classification_plot')
    plt.show()
    return (
        SVC,
        X,
        datasets,
        iris,
        np,
        plot_svc_decision_boundary,
        pred_1,
        pred_2,
        pred_3,
        svm_clf,
        x0,
        y,
    )


@app.cell
def _(SVC, np, plot_svc_decision_boundary, plt, save_fig):
    from sklearn.preprocessing import StandardScaler
    Xs = np.array([[1, 50], [5, 20], [3, 80], [5, 60]]).astype(np.float64)
    ys = np.array([0, 0, 1, 1])
    svm_clf_1 = SVC(kernel='linear', C=100).fit(Xs, ys)
    scaler = StandardScaler()
    _X_scaled = scaler.fit_transform(Xs)
    svm_clf_scaled = SVC(kernel='linear', C=100).fit(_X_scaled, ys)
    plt.figure(figsize=(9, 2.7))
    plt.subplot(121)
    plt.plot(Xs[:, 0][ys == 1], Xs[:, 1][ys == 1], 'bo')
    plt.plot(Xs[:, 0][ys == 0], Xs[:, 1][ys == 0], 'ms')
    plot_svc_decision_boundary(svm_clf_1, 0, 6)
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$\xa0\xa0\xa0\xa0', rotation=0)
    plt.title('Unscaled')
    plt.axis([0, 6, 0, 90])
    plt.grid()
    plt.subplot(122)
    plt.plot(_X_scaled[:, 0][ys == 1], _X_scaled[:, 1][ys == 1], 'bo')
    plt.plot(_X_scaled[:, 0][ys == 0], _X_scaled[:, 1][ys == 0], 'ms')
    plot_svc_decision_boundary(svm_clf_scaled, -2, 2)
    plt.xlabel("$x'_0$")
    plt.ylabel("$x'_1$  ", rotation=0)
    plt.title('Scaled')
    plt.axis([-2, 2, -2, 2])
    plt.grid()
    save_fig('sensitivity_to_feature_scales_plot')
    plt.show()
    return StandardScaler, Xs, scaler, svm_clf_1, svm_clf_scaled, ys


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Soft Margin Classification
        """
    )
    return


@app.cell
def _(SVC, X, np, plot_svc_decision_boundary, plt, save_fig, y):
    X_outliers = np.array([[3.4, 1.3], [3.2, 0.8]])
    y_outliers = np.array([0, 0])
    Xo1 = np.concatenate([X, X_outliers[:1]], axis=0)
    yo1 = np.concatenate([y, y_outliers[:1]], axis=0)
    Xo2 = np.concatenate([X, X_outliers[1:]], axis=0)
    yo2 = np.concatenate([y, y_outliers[1:]], axis=0)
    svm_clf2 = SVC(kernel='linear', C=10 ** 9)
    svm_clf2.fit(Xo2, yo2)
    _fig, _axes = plt.subplots(ncols=2, figsize=(10, 2.7), sharey=True)
    plt.sca(_axes[0])
    plt.plot(Xo1[:, 0][yo1 == 1], Xo1[:, 1][yo1 == 1], 'bs')
    plt.plot(Xo1[:, 0][yo1 == 0], Xo1[:, 1][yo1 == 0], 'yo')
    plt.text(0.3, 1.0, 'Impossible!', color='red', fontsize=18)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.annotate('Outlier', xy=(X_outliers[0][0], X_outliers[0][1]), xytext=(2.5, 1.7), ha='center', arrowprops=dict(facecolor='black', shrink=0.1))
    plt.axis([0, 5.5, 0, 2])
    plt.grid()
    plt.sca(_axes[1])
    plt.plot(Xo2[:, 0][yo2 == 1], Xo2[:, 1][yo2 == 1], 'bs')
    plt.plot(Xo2[:, 0][yo2 == 0], Xo2[:, 1][yo2 == 0], 'yo')
    plot_svc_decision_boundary(svm_clf2, 0, 5.5)
    plt.xlabel('Petal length')
    plt.annotate('Outlier', xy=(X_outliers[1][0], X_outliers[1][1]), xytext=(3.2, 0.08), ha='center', arrowprops=dict(facecolor='black', shrink=0.1))
    plt.axis([0, 5.5, 0, 2])
    plt.grid()
    save_fig('sensitivity_to_outliers_plot')
    plt.show()
    return X_outliers, Xo1, Xo2, svm_clf2, y_outliers, yo1, yo2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note: the default value for the `dual` hyperparameter of the `LinearSVC` and `LinearSVR` estimators will change from `True` to `"auto"` in Scikit-Learn 1.4, so I set `dual=True` throughout this notebook to ensure the output of this notebook remains unchanged.
        """
    )
    return


@app.cell
def _(StandardScaler):
    from sklearn.datasets import load_iris
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC
    iris_1 = load_iris(as_frame=True)
    X_1 = iris_1.data[['petal length (cm)', 'petal width (cm)']].values
    y_1 = iris_1.target == 2
    svm_clf_2 = make_pipeline(StandardScaler(), LinearSVC(C=1, dual=True, random_state=42))
    svm_clf_2.fit(X_1, y_1)
    return LinearSVC, X_1, iris_1, load_iris, make_pipeline, svm_clf_2, y_1


@app.cell
def _(svm_clf_2):
    X_new = [[5.5, 1.7], [5.0, 1.5]]
    svm_clf_2.predict(X_new)
    return (X_new,)


@app.cell
def _(X_new, svm_clf_2):
    svm_clf_2.decision_function(X_new)
    return


@app.cell
def _(
    LinearSVC,
    StandardScaler,
    X_1,
    make_pipeline,
    np,
    plot_svc_decision_boundary,
    plt,
    save_fig,
    y_1,
):
    scaler_1 = StandardScaler()
    svm_clf1 = LinearSVC(C=1, max_iter=10000, dual=True, random_state=42)
    svm_clf2_1 = LinearSVC(C=100, max_iter=10000, dual=True, random_state=42)
    scaled_svm_clf1 = make_pipeline(scaler_1, svm_clf1)
    scaled_svm_clf2 = make_pipeline(scaler_1, svm_clf2_1)
    scaled_svm_clf1.fit(X_1, y_1)
    scaled_svm_clf2.fit(X_1, y_1)
    b1 = svm_clf1.decision_function([-scaler_1.mean_ / scaler_1.scale_])
    b2 = svm_clf2_1.decision_function([-scaler_1.mean_ / scaler_1.scale_])
    w1 = svm_clf1.coef_[0] / scaler_1.scale_
    w2 = svm_clf2_1.coef_[0] / scaler_1.scale_
    svm_clf1.intercept_ = np.array([b1])
    svm_clf2_1.intercept_ = np.array([b2])
    svm_clf1.coef_ = np.array([w1])
    svm_clf2_1.coef_ = np.array([w2])
    _t = y_1 * 2 - 1
    support_vectors_idx1 = (_t * (X_1.dot(w1) + b1) < 1).ravel()
    support_vectors_idx2 = (_t * (X_1.dot(w2) + b2) < 1).ravel()
    svm_clf1.support_vectors_ = X_1[support_vectors_idx1]
    svm_clf2_1.support_vectors_ = X_1[support_vectors_idx2]
    _fig, _axes = plt.subplots(ncols=2, figsize=(10, 2.7), sharey=True)
    plt.sca(_axes[0])
    plt.plot(X_1[:, 0][y_1 == 1], X_1[:, 1][y_1 == 1], 'g^', label='Iris virginica')
    plt.plot(X_1[:, 0][y_1 == 0], X_1[:, 1][y_1 == 0], 'bs', label='Iris versicolor')
    plot_svc_decision_boundary(svm_clf1, 4, 5.9)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.legend(loc='upper left')
    plt.title(f'$C = {svm_clf1.C}$')
    plt.axis([4, 5.9, 0.8, 2.8])
    plt.grid()
    plt.sca(_axes[1])
    plt.plot(X_1[:, 0][y_1 == 1], X_1[:, 1][y_1 == 1], 'g^')
    plt.plot(X_1[:, 0][y_1 == 0], X_1[:, 1][y_1 == 0], 'bs')
    plot_svc_decision_boundary(svm_clf2_1, 4, 5.99)
    plt.xlabel('Petal length')
    plt.title(f'$C = {svm_clf2_1.C}$')
    plt.axis([4, 5.9, 0.8, 2.8])
    plt.grid()
    save_fig('regularization_plot')
    plt.show()
    return (
        b1,
        b2,
        scaled_svm_clf1,
        scaled_svm_clf2,
        scaler_1,
        support_vectors_idx1,
        support_vectors_idx2,
        svm_clf1,
        svm_clf2_1,
        w1,
        w2,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Nonlinear SVM Classification
        """
    )
    return


@app.cell
def _(np, plt, save_fig):
    X1D = np.linspace(-4, 4, 9).reshape(-1, 1)
    X2D = np.c_[X1D, X1D ** 2]
    y_2 = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])
    plt.figure(figsize=(10, 3))
    plt.subplot(121)
    plt.grid(True)
    plt.axhline(y=0, color='k')
    plt.plot(X1D[:, 0][y_2 == 0], np.zeros(4), 'bs')
    plt.plot(X1D[:, 0][y_2 == 1], np.zeros(5), 'g^')
    plt.gca().get_yaxis().set_ticks([])
    plt.xlabel('$x_1$')
    plt.axis([-4.5, 4.5, -0.2, 0.2])
    plt.subplot(122)
    plt.grid(True)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.plot(X2D[:, 0][y_2 == 0], X2D[:, 1][y_2 == 0], 'bs')
    plt.plot(X2D[:, 0][y_2 == 1], X2D[:, 1][y_2 == 1], 'g^')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$\xa0\xa0', rotation=0)
    plt.gca().get_yaxis().set_ticks([0, 4, 8, 12, 16])
    plt.plot([-4.5, 4.5], [6.5, 6.5], 'r--', linewidth=3)
    plt.axis([-4.5, 4.5, -1, 17])
    plt.subplots_adjust(right=1)
    save_fig('higher_dimensions_plot', tight_layout=False)
    plt.show()
    return X1D, X2D, y_2


@app.cell
def _(LinearSVC, StandardScaler, make_pipeline):
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import PolynomialFeatures
    X_2, y_3 = make_moons(n_samples=100, noise=0.15, random_state=42)
    polynomial_svm_clf = make_pipeline(PolynomialFeatures(degree=3), StandardScaler(), LinearSVC(C=10, max_iter=10000, dual=True, random_state=42))
    polynomial_svm_clf.fit(X_2, y_3)
    return PolynomialFeatures, X_2, make_moons, polynomial_svm_clf, y_3


@app.cell
def _(X_2, np, plt, polynomial_svm_clf, save_fig, y_3):
    def plot_dataset(X, y, axes):
        plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'bs')
        plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'g^')
        plt.axis(_axes)
        plt.grid(True)
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$', rotation=0)

    def plot_predictions(clf, axes):
        x0s = np.linspace(_axes[0], _axes[1], 100)
        x1s = np.linspace(_axes[2], _axes[3], 100)
        x0, x1 = np.meshgrid(x0s, x1s)
        X = np.c_[x0.ravel(), x1.ravel()]
        _y_pred = clf.predict(X).reshape(x0.shape)
        y_decision = clf.decision_function(X).reshape(x0.shape)
        plt.contourf(x0, x1, _y_pred, cmap=plt.cm.brg, alpha=0.2)
        plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)
    plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
    plot_dataset(X_2, y_3, [-1.5, 2.5, -1, 1.5])
    save_fig('moons_polynomial_svc_plot')
    plt.show()
    return plot_dataset, plot_predictions


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Polynomial Kernel
        """
    )
    return


@app.cell
def _(SVC, StandardScaler, X_2, make_pipeline, y_3):
    poly_kernel_svm_clf = make_pipeline(StandardScaler(), SVC(kernel='poly', degree=3, coef0=1, C=5))
    poly_kernel_svm_clf.fit(X_2, y_3)
    return (poly_kernel_svm_clf,)


@app.cell
def _(
    SVC,
    StandardScaler,
    X_2,
    make_pipeline,
    plot_dataset,
    plot_predictions,
    plt,
    poly_kernel_svm_clf,
    save_fig,
    y_3,
):
    poly100_kernel_svm_clf = make_pipeline(StandardScaler(), SVC(kernel='poly', degree=10, coef0=100, C=5))
    poly100_kernel_svm_clf.fit(X_2, y_3)
    _fig, _axes = plt.subplots(ncols=2, figsize=(10.5, 4), sharey=True)
    plt.sca(_axes[0])
    plot_predictions(poly_kernel_svm_clf, [-1.5, 2.45, -1, 1.5])
    plot_dataset(X_2, y_3, [-1.5, 2.4, -1, 1.5])
    plt.title('degree=3, coef0=1, C=5')
    plt.sca(_axes[1])
    plot_predictions(poly100_kernel_svm_clf, [-1.5, 2.45, -1, 1.5])
    plot_dataset(X_2, y_3, [-1.5, 2.4, -1, 1.5])
    plt.title('degree=10, coef0=100, C=5')
    plt.ylabel('')
    save_fig('moons_kernelized_polynomial_svc_plot')
    plt.show()
    return (poly100_kernel_svm_clf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Similarity Features
        """
    )
    return


@app.cell
def _(X1D, np, plt, save_fig):
    def gaussian_rbf(x, landmark, gamma):
        return np.exp(-_gamma * np.linalg.norm(x - landmark, axis=1) ** 2)
    _gamma = 0.3
    x1s = np.linspace(-4.5, 4.5, 200).reshape(-1, 1)
    x2s = gaussian_rbf(x1s, -2, _gamma)
    x3s = gaussian_rbf(x1s, 1, _gamma)
    XK = np.c_[gaussian_rbf(X1D, -2, _gamma), gaussian_rbf(X1D, 1, _gamma)]
    yk = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])
    plt.figure(figsize=(10.5, 4))
    plt.subplot(121)
    plt.grid(True)
    plt.axhline(y=0, color='k')
    plt.scatter(x=[-2, 1], y=[0, 0], s=150, alpha=0.5, c='red')
    plt.plot(X1D[:, 0][yk == 0], np.zeros(4), 'bs')
    plt.plot(X1D[:, 0][yk == 1], np.zeros(5), 'g^')
    plt.plot(x1s, x2s, 'g--')
    plt.plot(x1s, x3s, 'b:')
    plt.gca().get_yaxis().set_ticks([0, 0.25, 0.5, 0.75, 1])
    plt.xlabel('$x_1$')
    plt.ylabel('Similarity')
    plt.annotate('$\\mathbf{x}$', xy=(X1D[3, 0], 0), xytext=(-0.5, 0.2), ha='center', arrowprops=dict(facecolor='black', shrink=0.1), fontsize=16)
    plt.text(-2, 0.9, '$x_2$', ha='center', fontsize=15)
    plt.text(1, 0.9, '$x_3$', ha='center', fontsize=15)
    plt.axis([-4.5, 4.5, -0.1, 1.1])
    plt.subplot(122)
    plt.grid(True)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.plot(XK[:, 0][yk == 0], XK[:, 1][yk == 0], 'bs')
    plt.plot(XK[:, 0][yk == 1], XK[:, 1][yk == 1], 'g^')
    plt.xlabel('$x_2$')
    plt.ylabel('$x_3$\xa0\xa0', rotation=0)
    plt.annotate('$\\phi\\left(\\mathbf{x}\\right)$', xy=(XK[3, 0], XK[3, 1]), xytext=(0.65, 0.5), ha='center', arrowprops=dict(facecolor='black', shrink=0.1), fontsize=16)
    plt.plot([-0.1, 1.1], [0.57, -0.1], 'r--', linewidth=3)
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.subplots_adjust(right=1)
    save_fig('kernel_method_plot')
    plt.show()
    return XK, gaussian_rbf, x1s, x2s, x3s, yk


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Gaussian RBF Kernel
        """
    )
    return


@app.cell
def _(SVC, StandardScaler, X_2, make_pipeline, y_3):
    _rbf_kernel_svm_clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma=5, C=0.001))
    _rbf_kernel_svm_clf.fit(X_2, y_3)
    return


@app.cell
def _(
    SVC,
    StandardScaler,
    X_2,
    make_pipeline,
    plot_dataset,
    plot_predictions,
    plt,
    save_fig,
    y_3,
):
    gamma1, gamma2 = (0.1, 5)
    C1, C2 = (0.001, 1000)
    hyperparams = ((gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2))
    svm_clfs = []
    for _gamma, C in hyperparams:
        _rbf_kernel_svm_clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma=_gamma, C=C))
        _rbf_kernel_svm_clf.fit(X_2, y_3)
        svm_clfs.append(_rbf_kernel_svm_clf)
    _fig, _axes = plt.subplots(nrows=2, ncols=2, figsize=(10.5, 7), sharex=True, sharey=True)
    for i, svm_clf_3 in enumerate(svm_clfs):
        plt.sca(_axes[i // 2, i % 2])
        plot_predictions(svm_clf_3, [-1.5, 2.45, -1, 1.5])
        plot_dataset(X_2, y_3, [-1.5, 2.45, -1, 1.5])
        _gamma, C = hyperparams[i]
        plt.title(f'gamma={_gamma}, C={C}')
        if i in (0, 1):
            plt.xlabel('')
        if i in (1, 3):
            plt.ylabel('')
    save_fig('moons_rbf_svc_plot')
    plt.show()
    return C, C1, C2, gamma1, gamma2, hyperparams, i, svm_clf_3, svm_clfs


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # SVM Regression
        """
    )
    return


@app.cell
def _(StandardScaler, make_pipeline, np):
    from sklearn.svm import LinearSVR
    np.random.seed(42)
    X_3 = 2 * np.random.rand(50, 1)
    y_4 = 4 + 3 * X_3[:, 0] + np.random.randn(50)
    svm_reg = make_pipeline(StandardScaler(), LinearSVR(epsilon=0.5, dual=True, random_state=42))
    svm_reg.fit(X_3, y_4)
    return LinearSVR, X_3, svm_reg, y_4


@app.cell
def _(
    LinearSVR,
    StandardScaler,
    X_3,
    make_pipeline,
    np,
    plt,
    save_fig,
    svm_reg,
    y_4,
):
    def find_support_vectors(svm_reg, X, y):
        _y_pred = svm_reg.predict(X)
        epsilon = svm_reg[-1].epsilon
        off_margin = np.abs(y - _y_pred) >= epsilon
        return np.argwhere(off_margin)

    def plot_svm_regression(svm_reg, X, y, axes):
        x1s = np.linspace(_axes[0], _axes[1], 100).reshape(100, 1)
        _y_pred = svm_reg.predict(x1s)
        epsilon = svm_reg[-1].epsilon
        plt.plot(x1s, _y_pred, 'k-', linewidth=2, label='$\\hat{y}$', zorder=-2)
        plt.plot(x1s, _y_pred + epsilon, 'k--', zorder=-2)
        plt.plot(x1s, _y_pred - epsilon, 'k--', zorder=-2)
        plt.scatter(X[svm_reg._support], y[svm_reg._support], s=180, facecolors='#AAA', zorder=-1)
        plt.plot(X, y, 'bo')
        plt.xlabel('$x_1$')
        plt.legend(loc='upper left')
        plt.axis(_axes)
    svm_reg2 = make_pipeline(StandardScaler(), LinearSVR(epsilon=1.2, dual=True, random_state=42))
    svm_reg2.fit(X_3, y_4)
    svm_reg._support = find_support_vectors(svm_reg, X_3, y_4)
    svm_reg2._support = find_support_vectors(svm_reg2, X_3, y_4)
    eps_x1 = 1
    eps_y_pred = svm_reg2.predict([[eps_x1]])
    _fig, _axes = plt.subplots(ncols=2, figsize=(9, 4), sharey=True)
    plt.sca(_axes[0])
    plot_svm_regression(svm_reg, X_3, y_4, [0, 2, 3, 11])
    plt.title(f'epsilon={svm_reg[-1].epsilon}')
    plt.ylabel('$y$', rotation=0)
    plt.grid()
    plt.sca(_axes[1])
    plot_svm_regression(svm_reg2, X_3, y_4, [0, 2, 3, 11])
    plt.title(f'epsilon={svm_reg2[-1].epsilon}')
    plt.annotate('', xy=(eps_x1, eps_y_pred), xycoords='data', xytext=(eps_x1, eps_y_pred - svm_reg2[-1].epsilon), textcoords='data', arrowprops={'arrowstyle': '<->', 'linewidth': 1.5})
    plt.text(0.9, 5.4, '$\\epsilon$', fontsize=16)
    plt.grid()
    save_fig('svm_regression_plot')
    plt.show()
    return (
        eps_x1,
        eps_y_pred,
        find_support_vectors,
        plot_svm_regression,
        svm_reg2,
    )


@app.cell
def _(StandardScaler, make_pipeline, np):
    from sklearn.svm import SVR
    np.random.seed(42)
    X_4 = 2 * np.random.rand(50, 1) - 1
    y_5 = 0.2 + 0.1 * X_4[:, 0] + 0.5 * X_4[:, 0] ** 2 + np.random.randn(50) / 10
    svm_poly_reg = make_pipeline(StandardScaler(), SVR(kernel='poly', degree=2, C=0.01, epsilon=0.1))
    svm_poly_reg.fit(X_4, y_5)
    return SVR, X_4, svm_poly_reg, y_5


@app.cell
def _(
    SVR,
    StandardScaler,
    X_4,
    find_support_vectors,
    make_pipeline,
    plot_svm_regression,
    plt,
    save_fig,
    svm_poly_reg,
    y_5,
):
    svm_poly_reg2 = make_pipeline(StandardScaler(), SVR(kernel='poly', degree=2, C=100))
    svm_poly_reg2.fit(X_4, y_5)
    svm_poly_reg._support = find_support_vectors(svm_poly_reg, X_4, y_5)
    svm_poly_reg2._support = find_support_vectors(svm_poly_reg2, X_4, y_5)
    _fig, _axes = plt.subplots(ncols=2, figsize=(9, 4), sharey=True)
    plt.sca(_axes[0])
    plot_svm_regression(svm_poly_reg, X_4, y_5, [-1, 1, 0, 1])
    plt.title(f'degree={svm_poly_reg[-1].degree}, C={svm_poly_reg[-1].C}, epsilon={svm_poly_reg[-1].epsilon}')
    plt.ylabel('$y$', rotation=0)
    plt.grid()
    plt.sca(_axes[1])
    plot_svm_regression(svm_poly_reg2, X_4, y_5, [-1, 1, 0, 1])
    plt.title(f'degree={svm_poly_reg2[-1].degree}, C={svm_poly_reg2[-1].C}, epsilon={svm_poly_reg2[-1].epsilon}')
    plt.grid()
    save_fig('svm_with_polynomial_kernel_plot')
    plt.show()
    return (svm_poly_reg2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Under the hood
        """
    )
    return


@app.cell
def _(np, plt, save_fig):
    import matplotlib.patches as patches

    def plot_2D_decision_function(w, b, ylabel=True, x1_lim=[-3, 3]):
        x1 = np.linspace(x1_lim[0], x1_lim[1], 200)
        y = w * x1 + b
        half_margin = 1 / w
        plt.plot(x1, y, 'b-', linewidth=2, label='$s = w_1 x_1$')
        plt.axhline(y=0, color='k', linewidth=1)
        plt.axvline(x=0, color='k', linewidth=1)
        rect = patches.Rectangle((-half_margin, -2), 2 * half_margin, 4, edgecolor='none', facecolor='gray', alpha=0.2)
        plt.gca().add_patch(rect)
        plt.plot([-3, 3], [1, 1], 'k--', linewidth=1)
        plt.plot([-3, 3], [-1, -1], 'k--', linewidth=1)
        plt.plot(half_margin, 1, 'k.')
        plt.plot(-half_margin, -1, 'k.')
        plt.axis(x1_lim + [-2, 2])
        plt.xlabel('$x_1$')
        if ylabel:
            plt.ylabel('$s$', rotation=0, labelpad=5)
            plt.legend()
            plt.text(1.02, -1.6, 'Margin', ha='left', va='center', color='k')
        plt.annotate('', xy=(-half_margin, -1.6), xytext=(half_margin, -1.6), arrowprops={'ec': 'k', 'arrowstyle': '<->', 'linewidth': 1.5})
        plt.title(f'$w_1 = {w}$')
    _fig, _axes = plt.subplots(ncols=2, figsize=(9, 3.2), sharey=True)
    plt.sca(_axes[0])
    plot_2D_decision_function(1, 0)
    plt.grid()
    plt.sca(_axes[1])
    plot_2D_decision_function(0.5, 0, ylabel=False)
    plt.grid()
    save_fig('small_w_large_margin_plot')
    plt.show()
    return patches, plot_2D_decision_function


@app.cell
def _(np, plt, save_fig):
    # extra code – this cell generates and saves Figure 5–13

    s = np.linspace(-2.5, 2.5, 200)
    hinge_pos = np.where(1 - s < 0, 0, 1 - s)  # max(0, 1 - s)
    hinge_neg = np.where(1 + s < 0, 0, 1 + s)  # max(0, 1 + s)

    titles = (r"Hinge loss = $max(0, 1 - s\,t)$", "Squared Hinge loss")

    fix, axs = plt.subplots(1, 2, sharey=True, figsize=(8.2, 3))

    for ax, loss_pos, loss_neg, title in zip(
            axs, (hinge_pos, hinge_pos ** 2), (hinge_neg, hinge_neg ** 2), titles):
        ax.plot(s, loss_pos, "g-", linewidth=2, zorder=10, label="$t=1$")
        ax.plot(s, loss_neg, "r--", linewidth=2, zorder=10, label="$t=-1$")
        ax.grid(True)
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.set_xlabel(r"$s = \mathbf{w}^\intercal \mathbf{x} + b$")
        ax.axis([-2.5, 2.5, -0.5, 2.5])
        ax.legend(loc="center right")
        ax.set_title(title)
        ax.set_yticks(np.arange(0, 2.5, 1))
        ax.set_aspect("equal")

    save_fig("hinge_plot")
    plt.show()
    return (
        ax,
        axs,
        fix,
        hinge_neg,
        hinge_pos,
        loss_neg,
        loss_pos,
        s,
        title,
        titles,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Extra Material
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Linear SVM classifier implementation using Batch Gradient Descent
        """
    )
    return


@app.cell
def _(iris_1):
    X_5 = iris_1.data[['petal length (cm)', 'petal width (cm)']].values
    y_6 = iris_1.target == 2
    return X_5, y_6


@app.cell
def _(np):
    from sklearn.base import BaseEstimator

    class MyLinearSVC(BaseEstimator):

        def __init__(self, C=1, eta0=1, eta_d=10000, n_epochs=1000, random_state=None):
            self.C = C
            self.eta0 = eta0
            self.n_epochs = n_epochs
            self.random_state = random_state
            self.eta_d = eta_d

        def eta(self, epoch):
            return self.eta0 / (epoch + self.eta_d)

        def fit(self, X, y):
            if self.random_state:
                np.random.seed(self.random_state)
            w = np.random.randn(X.shape[1], 1)
            b = 0
            _t = np.array(y, dtype=np.float64).reshape(-1, 1) * 2 - 1
            X_t = X * _t
            self.Js = []
            for epoch in range(self.n_epochs):
                support_vectors_idx = (X_t.dot(w) + _t * b < 1).ravel()
                X_t_sv = X_t[support_vectors_idx]
                t_sv = _t[support_vectors_idx]
                J = 1 / 2 * (w * w).sum() + self.C * ((1 - X_t_sv.dot(w)).sum() - b * t_sv.sum())
                self.Js.append(J)
                w_gradient_vector = w - self.C * X_t_sv.sum(axis=0).reshape(-1, 1)
                b_derivative = -self.C * t_sv.sum()
                w = w - self.eta(epoch) * w_gradient_vector
                b = b - self.eta(epoch) * b_derivative
            self.intercept_ = np.array([b])
            self.coef_ = np.array([w])
            support_vectors_idx = (X_t.dot(w) + _t * b < 1).ravel()
            self.support_vectors_ = X[support_vectors_idx]
            return self

        def decision_function(self, X):
            return X.dot(self.coef_[0]) + self.intercept_[0]

        def predict(self, X):
            return self.decision_function(X) >= 0
    return BaseEstimator, MyLinearSVC


@app.cell
def _(MyLinearSVC, X_5, np, y_6):
    C_1 = 2
    svm_clf_4 = MyLinearSVC(C=C_1, eta0=10, eta_d=1000, n_epochs=60000, random_state=2)
    svm_clf_4.fit(X_5, y_6)
    svm_clf_4.predict(np.array([[5, 2], [4, 1]]))
    return C_1, svm_clf_4


@app.cell
def _(plt, svm_clf_4):
    plt.plot(range(svm_clf_4.n_epochs), svm_clf_4.Js)
    plt.axis([0, svm_clf_4.n_epochs, 0, 100])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()
    return


@app.cell
def _(svm_clf_4):
    print(svm_clf_4.intercept_, svm_clf_4.coef_)
    return


@app.cell
def _(C_1, SVC, X_5, y_6):
    svm_clf2_2 = SVC(kernel='linear', C=C_1)
    svm_clf2_2.fit(X_5, y_6.ravel())
    print(svm_clf2_2.intercept_, svm_clf2_2.coef_)
    return (svm_clf2_2,)


@app.cell
def _(X_5, plot_svc_decision_boundary, plt, svm_clf2_2, svm_clf_4, y_6):
    yr = y_6.ravel()
    _fig, _axes = plt.subplots(ncols=2, figsize=(11, 3.2), sharey=True)
    plt.sca(_axes[0])
    plt.plot(X_5[:, 0][yr == 1], X_5[:, 1][yr == 1], 'g^', label='Iris virginica')
    plt.plot(X_5[:, 0][yr == 0], X_5[:, 1][yr == 0], 'bs', label='Not Iris virginica')
    plot_svc_decision_boundary(svm_clf_4, 4, 6)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.title('MyLinearSVC')
    plt.axis([4, 6, 0.8, 2.8])
    plt.legend(loc='upper left')
    plt.grid()
    plt.sca(_axes[1])
    plt.plot(X_5[:, 0][yr == 1], X_5[:, 1][yr == 1], 'g^')
    plt.plot(X_5[:, 0][yr == 0], X_5[:, 1][yr == 0], 'bs')
    plot_svc_decision_boundary(svm_clf2_2, 4, 6)
    plt.xlabel('Petal length')
    plt.title('SVC')
    plt.axis([4, 6, 0.8, 2.8])
    plt.grid()
    plt.show()
    return (yr,)


@app.cell
def _(C_1, X_5, np, plot_svc_decision_boundary, plt, y_6, yr):
    from sklearn.linear_model import SGDClassifier
    sgd_clf = SGDClassifier(loss='hinge', alpha=0.017, max_iter=1000, tol=0.001, random_state=42)
    sgd_clf.fit(X_5, y_6)
    m = len(X_5)
    _t = np.array(y_6).reshape(-1, 1) * 2 - 1
    X_b = np.c_[np.ones((m, 1)), X_5]
    X_b_t = X_b * _t
    sgd_theta = np.r_[sgd_clf.intercept_[0], sgd_clf.coef_[0]]
    print(sgd_theta)
    support_vectors_idx = (X_b_t.dot(sgd_theta) < 1).ravel()
    sgd_clf.support_vectors_ = X_5[support_vectors_idx]
    sgd_clf.C = C_1
    plt.figure(figsize=(5.5, 3.2))
    plt.plot(X_5[:, 0][yr == 1], X_5[:, 1][yr == 1], 'g^')
    plt.plot(X_5[:, 0][yr == 0], X_5[:, 1][yr == 0], 'bs')
    plot_svc_decision_boundary(sgd_clf, 4, 6)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.title('SGDClassifier')
    plt.axis([4, 6, 0.8, 2.8])
    plt.grid()
    plt.show()
    return (
        SGDClassifier,
        X_b,
        X_b_t,
        m,
        sgd_clf,
        sgd_theta,
        support_vectors_idx,
    )


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
        ## 1. to 8.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        1. The fundamental idea behind Support Vector Machines is to fit the widest possible "street" between the classes. In other words, the goal is to have the largest possible margin between the decision boundary that separates the two classes and the training instances. When performing soft margin classification, the SVM searches for a compromise between perfectly separating the two classes and having the widest possible street (i.e., a few instances may end up on the street). Another key idea is to use kernels when training on nonlinear datasets. SVMs can also be tweaked to perform linear and nonlinear regression, as well as novelty detection.
        2. After training an SVM, a _support vector_ is any instance located on the "street" (see the previous answer), including its border. The decision boundary is entirely determined by the support vectors. Any instance that is _not_ a support vector (i.e., is off the street) has no influence whatsoever; you could remove them, add more instances, or move them around, and as long as they stay off the street they won't affect the decision boundary. Computing the predictions with a kernelized SVM only involves the support vectors, not the whole training set.
        3. SVMs try to fit the largest possible "street" between the classes (see the first answer), so if the training set is not scaled, the SVM will tend to neglect small features (see Figure 5–2).
        4. You can use the `decision_function()` method to get confidence scores. These scores represent the distance between the instance and the decision boundary. However, they cannot be directly converted into an estimation of the class probability. If you set `probability=True` when creating an `SVC`, then at the end of training it will use 5-fold cross-validation to generate out-of-sample scores for the training samples, and it will train a `LogisticRegression` model to map these scores to estimated probabilities. The `predict_proba()` and `predict_log_proba()` methods will then be available.
        5. All three classes can be used for large-margin linear classification. The `SVC` class also supports the kernel trick, which makes it capable of handling nonlinear tasks. However, this comes at a cost: the `SVC` class does not scale well to datasets with many instances. It does scale well to a large number of features, though. The `LinearSVC` class implements an optimized algorithm for linear SVMs, while `SGDClassifier` uses Stochastic Gradient Descent. Depending on the dataset `LinearSVC` may be a bit faster than `SGDClassifier`, but not always, and `SGDClassifier` is more flexible, plus it supports incremental learning.
        6. If an SVM classifier trained with an RBF kernel underfits the training set, there might be too much regularization. To decrease it, you need to increase `gamma` or `C` (or both).
        7. A Regression SVM model tries to fit as many instances within a small margin around its predictions. If you add instances within this margin, the model will not be affected at all: it is said to be _ϵ-insensitive_.
        8. The kernel trick is mathematical technique that makes it possible to train a nonlinear SVM model. The resulting model is equivalent to mapping the inputs to another space using a nonlinear transformation, then training a linear SVM on the resulting high-dimensional inputs. The kernel trick gives the same result without having to transform the inputs at all.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 9.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        _Exercise: Train a `LinearSVC` on a linearly separable dataset. Then train an `SVC` and a `SGDClassifier` on the same dataset. See if you can get them to produce roughly the same model._
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's use the Iris dataset: the Iris Setosa and Iris Versicolor classes are linearly separable.
        """
    )
    return


@app.cell
def _(datasets):
    iris_2 = datasets.load_iris(as_frame=True)
    X_6 = iris_2.data[['petal length (cm)', 'petal width (cm)']].values
    y_7 = iris_2.target
    _setosa_or_versicolor = (y_7 == 0) | (y_7 == 1)
    X_6 = X_6[_setosa_or_versicolor]
    y_7 = y_7[_setosa_or_versicolor]
    return X_6, iris_2, y_7


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's build and train 3 models:
        * Remember that `LinearSVC` uses `loss="squared_hinge"` by default, so if we want all 3 models to produce similar results, we need to set `loss="hinge"`.
        * Also, the `SVC` class uses an RBF kernel by default, so we need to set `kernel="linear"` to get similar results as the other two models.
        * Lastly, the `SGDClassifier` class does not have a `C` hyperparameter, but it has another regularization hyperparameter called `alpha`, so we can tweak it to get similar results as the other two models.
        """
    )
    return


@app.cell
def _(LinearSVC_1, SGDClassifier, SVC_1, StandardScaler, X_6, y_7):
    from sklearn.svm import SVC, LinearSVC
    C_2 = 5
    alpha = 0.05
    scaler_2 = StandardScaler()
    _X_scaled = scaler_2.fit_transform(X_6)
    lin_clf = LinearSVC_1(loss='hinge', C=C_2, dual=True, random_state=42).fit(_X_scaled, y_7)
    svc_clf = SVC_1(kernel='linear', C=C_2).fit(_X_scaled, y_7)
    sgd_clf_1 = SGDClassifier(alpha=alpha, random_state=42).fit(_X_scaled, y_7)
    return C_2, LinearSVC, SVC, alpha, lin_clf, scaler_2, sgd_clf_1, svc_clf


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's plot the decision boundaries of these three models:
        """
    )
    return


@app.cell
def _(X_6, lin_clf, plt, scaler_2, sgd_clf_1, svc_clf, y_7):
    def compute_decision_boundary(model):
        w = -model.coef_[0, 0] / model.coef_[0, 1]
        b = -model.intercept_[0] / model.coef_[0, 1]
        return scaler_2.inverse_transform([[-10, -10 * w + b], [10, 10 * w + b]])
    lin_line = compute_decision_boundary(lin_clf)
    svc_line = compute_decision_boundary(svc_clf)
    sgd_line = compute_decision_boundary(sgd_clf_1)
    plt.figure(figsize=(11, 4))
    plt.plot(lin_line[:, 0], lin_line[:, 1], 'k:', label='LinearSVC')
    plt.plot(svc_line[:, 0], svc_line[:, 1], 'b--', linewidth=2, label='SVC')
    plt.plot(sgd_line[:, 0], sgd_line[:, 1], 'r-', label='SGDClassifier')
    plt.plot(X_6[:, 0][y_7 == 1], X_6[:, 1][y_7 == 1], 'bs')
    plt.plot(X_6[:, 0][y_7 == 0], X_6[:, 1][y_7 == 0], 'yo')
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.legend(loc='upper center')
    plt.axis([0, 5.5, 0, 2])
    plt.grid()
    plt.show()
    return compute_decision_boundary, lin_line, sgd_line, svc_line


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Close enough!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 10.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        _Exercise: Train an SVM classifier on the Wine dataset, which you can load using `sklearn.datasets.load_wine()`. This dataset contains the chemical analysis of 178 wine samples produced by 3 different cultivators: the goal is to train a classification model capable of predicting the cultivator based on the wine's chemical analysis. Since SVM classifiers are binary classifiers, you will need to use one-versus-all to classify all 3 classes. What accuracy can you reach?_
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        First, let's fetch the dataset, look at its description, then split it into a training set and a test set:
        """
    )
    return


@app.cell
def _():
    from sklearn.datasets import load_wine

    wine = load_wine(as_frame=True)
    return load_wine, wine


@app.cell
def _(wine):
    print(wine.DESCR)
    return


@app.cell
def _(wine):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, random_state=42)
    return X_test, X_train, train_test_split, y_test, y_train


@app.cell
def _(X_train):
    X_train.head()
    return


@app.cell
def _(y_train):
    y_train.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's start simple, with a linear SVM classifier. It will automatically use the One-vs-All (also called One-vs-the-Rest, OvR) strategy, so there's nothing special we need to do to handle multiple classes. Easy, right?
        """
    )
    return


@app.cell
def _(LinearSVC_1, X_train, y_train):
    lin_clf_1 = LinearSVC_1(dual=True, random_state=42)
    lin_clf_1.fit(X_train, y_train)
    return (lin_clf_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Oh no! It failed to converge. Can you guess why? Do you think we must just increase the number of training iterations? Let's see:
        """
    )
    return


@app.cell
def _(LinearSVC_1, X_train, y_train):
    lin_clf_2 = LinearSVC_1(max_iter=1000000, dual=True, random_state=42)
    lin_clf_2.fit(X_train, y_train)
    return (lin_clf_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Even with one million iterations, it still did not converge. There must be another problem.

        Let's still evaluate this model with `cross_val_score`, it will serve as a baseline:
        """
    )
    return


@app.cell
def _(X_train, lin_clf_2, y_train):
    from sklearn.model_selection import cross_val_score
    cross_val_score(lin_clf_2, X_train, y_train).mean()
    return (cross_val_score,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Well 91% accuracy on this dataset is not great. So did you guess what the problem is?

        That's right, we forgot to scale the features! Always remember to scale the features when using SVMs:
        """
    )
    return


@app.cell
def _(LinearSVC_1, StandardScaler, X_train, make_pipeline, y_train):
    lin_clf_3 = make_pipeline(StandardScaler(), LinearSVC_1(dual=True, random_state=42))
    lin_clf_3.fit(X_train, y_train)
    return (lin_clf_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now it converges without any problem. Let's measure its performance:
        """
    )
    return


@app.cell
def _(X_train, cross_val_score, lin_clf_3, y_train):
    cross_val_score(lin_clf_3, X_train, y_train).mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Nice! We get 97.7% accuracy, that's much better.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's see if a kernelized SVM will do better. We will use a default `SVC` for now:
        """
    )
    return


@app.cell
def _(SVC_1, StandardScaler, X_train, cross_val_score, make_pipeline, y_train):
    svm_clf_5 = make_pipeline(StandardScaler(), SVC_1(random_state=42))
    cross_val_score(svm_clf_5, X_train, y_train).mean()
    return (svm_clf_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        That's not better, but perhaps we need to do a bit of hyperparameter tuning:
        """
    )
    return


@app.cell
def _(X_train, svm_clf_5, y_train):
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import loguniform, uniform
    _param_distrib = {'svc__gamma': loguniform(0.001, 0.1), 'svc__C': uniform(1, 10)}
    rnd_search_cv = RandomizedSearchCV(svm_clf_5, _param_distrib, n_iter=100, cv=5, random_state=42)
    rnd_search_cv.fit(X_train, y_train)
    rnd_search_cv.best_estimator_
    return RandomizedSearchCV, loguniform, rnd_search_cv, uniform


@app.cell
def _(rnd_search_cv):
    rnd_search_cv.best_score_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Ah, this looks excellent! Let's select this model. Now we can test it on the test set:
        """
    )
    return


@app.cell
def _(X_test, rnd_search_cv, y_test):
    rnd_search_cv.score(X_test, y_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This tuned kernelized SVM performs better than the `LinearSVC` model, but we get a lower score on the test set than we measured using cross-validation. This is quite common: since we did so much hyperparameter tuning, we ended up slightly overfitting the cross-validation test sets. It's tempting to tweak the hyperparameters a bit more until we get a better result on the test set, but this would probably not help, as we would just start overfitting the test set. Anyway, this score is not bad at all, so let's stop here.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 11.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        _Exercise: Train and fine-tune an SVM regressor on the California housing dataset. You can use the original dataset rather than the tweaked version we used in Chapter 2. The original dataset can be fetched using `sklearn.datasets.fetch_california_housing()`. The targets represent hundreds of thousands of dollars. Since there are over 20,000 instances, SVMs can be slow, so for hyperparameter tuning you should use much less instances (e.g., 2,000), to test many more hyperparameter combinations. What is your best model's RMSE?_
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's load the dataset:
        """
    )
    return


@app.cell
def _():
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    X_7 = housing.data
    y_8 = housing.target
    return X_7, fetch_california_housing, housing, y_8


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Split it into a training set and a test set:
        """
    )
    return


@app.cell
def _(X_7, train_test_split, y_8):
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_7, y_8, test_size=0.2, random_state=42)
    return X_test_1, X_train_1, y_test_1, y_train_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Don't forget to scale the data!
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's train a simple `LinearSVR` first:
        """
    )
    return


@app.cell
def _(LinearSVR, StandardScaler, X_train_1, make_pipeline, y_train_1):
    lin_svr = make_pipeline(StandardScaler(), LinearSVR(dual=True, random_state=42))
    lin_svr.fit(X_train_1, y_train_1)
    return (lin_svr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It did not converge, so let's increase `max_iter`:
        """
    )
    return


@app.cell
def _(LinearSVR, StandardScaler, X_train_1, make_pipeline, y_train_1):
    lin_svr_1 = make_pipeline(StandardScaler(), LinearSVR(max_iter=5000, dual=True, random_state=42))
    lin_svr_1.fit(X_train_1, y_train_1)
    return (lin_svr_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's see how it performs on the training set:
        """
    )
    return


@app.cell
def _(X_train_1, lin_svr_1, y_train_1):
    from sklearn.metrics import mean_squared_error
    _y_pred = lin_svr_1.predict(X_train_1)
    mse = mean_squared_error(y_train_1, _y_pred)
    mse
    return mean_squared_error, mse


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's look at the RMSE:
        """
    )
    return


@app.cell
def _(mse, np):
    np.sqrt(mse)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this dataset, the targets represent hundreds of thousands of dollars. The RMSE gives a rough idea of the kind of error you should expect (with a higher weight for large errors): so with this model we can expect errors close to $98,000! Not great. Let's see if we can do better with an RBF Kernel. We will use randomized search with cross validation to find the appropriate hyperparameter values for `C` and `gamma`:
        """
    )
    return


@app.cell
def _(
    RandomizedSearchCV,
    SVR,
    StandardScaler,
    X_train_1,
    loguniform,
    make_pipeline,
    uniform,
    y_train_1,
):
    svm_reg_1 = make_pipeline(StandardScaler(), SVR())
    _param_distrib = {'svr__gamma': loguniform(0.001, 0.1), 'svr__C': uniform(1, 10)}
    rnd_search_cv_1 = RandomizedSearchCV(svm_reg_1, _param_distrib, n_iter=100, cv=3, random_state=42)
    rnd_search_cv_1.fit(X_train_1[:2000], y_train_1[:2000])
    return rnd_search_cv_1, svm_reg_1


@app.cell
def _(rnd_search_cv_1):
    rnd_search_cv_1.best_estimator_
    return


@app.cell
def _(X_train_1, cross_val_score, rnd_search_cv_1, y_train_1):
    -cross_val_score(rnd_search_cv_1.best_estimator_, X_train_1, y_train_1, scoring='neg_root_mean_squared_error')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Looks much better than the linear model. Let's select this model and evaluate it on the test set:
        """
    )
    return


@app.cell
def _(X_test_1, mean_squared_error, rnd_search_cv_1, y_test_1):
    _y_pred = rnd_search_cv_1.best_estimator_.predict(X_test_1)
    rmse = mean_squared_error(y_test_1, _y_pred, squared=False)
    rmse
    return (rmse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        So SVMs worked very well on the Wine dataset, but not so much on the California Housing dataset. In Chapter 2, we found that Random Forests worked better for that dataset.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And that's all for today!
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

