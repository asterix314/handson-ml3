import marimo

__generated_with = "0.11.20"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Chapter 4 – Training Models**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        _This notebook contains all the sample code and solutions to the exercises in chapter 4._
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <table align="left">
          <td>
            <a href="https://colab.research.google.com/github/ageron/handson-ml3/blob/main/04_training_linear_models.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
          </td>
          <td>
            <a target="_blank" href="https://kaggle.com/kernels/welcome?src=https://github.com/ageron/handson-ml3/blob/main/04_training_linear_models.ipynb"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>
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
        And let's create the `images/training_linear_models` folder (if it doesn't already exist), and define the `save_fig()` function which is used through this notebook to save the figures in high-res for the book:
        """
    )
    return


@app.cell
def _(plt):
    from pathlib import Path

    IMAGES_PATH = Path() / "images" / "training_linear_models"
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
        # Linear Regression
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## The Normal Equation
        """
    )
    return


@app.cell
def _():
    import numpy as np

    np.random.seed(42)  # to make this code example reproducible
    m = 100  # number of instances
    X = 2 * np.random.rand(m, 1)  # column vector
    y = 4 + 3 * X + np.random.randn(m, 1)  # column vector
    return X, m, np, y


@app.cell
def _(X, plt, save_fig, y):
    # extra code – generates and saves Figure 4–1


    plt.figure(figsize=(6, 4))
    plt.plot(X, y, "b.")
    plt.xlabel("$x_1$")
    plt.ylabel("$y$", rotation=0)
    plt.axis([0, 2, 0, 15])
    plt.grid()
    save_fig("generated_data_plot")
    plt.show()
    return


@app.cell
def _(X, np, y):
    from sklearn.preprocessing import add_dummy_feature

    X_b = add_dummy_feature(X)  # add x0 = 1 to each instance
    theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return X_b, add_dummy_feature, theta_best


@app.cell
def _(theta_best):
    theta_best
    return


@app.cell
def _(add_dummy_feature, np, theta_best):
    X_new = np.array([[0], [2]])
    X_new_b = add_dummy_feature(X_new)  # add x0 = 1 to each instance
    y_predict = X_new_b @ theta_best
    y_predict
    return X_new, X_new_b, y_predict


@app.cell
def _(X, X_new, plt, save_fig, y, y_predict):
    plt.figure(figsize=(6, 4))  # extra code – not needed, just formatting
    plt.plot(X_new, y_predict, "r-", label="Predictions")
    plt.plot(X, y, "b.")

    # extra code – beautifies and saves Figure 4–2
    plt.xlabel("$x_1$")
    plt.ylabel("$y$", rotation=0)
    plt.axis([0, 2, 0, 15])
    plt.grid()
    plt.legend(loc="upper left")
    save_fig("linear_model_predictions_plot")

    plt.show()
    return


@app.cell
def _(X, y):
    from sklearn.linear_model import LinearRegression

    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    lin_reg.intercept_, lin_reg.coef_
    return LinearRegression, lin_reg


@app.cell
def _(X_new, lin_reg):
    lin_reg.predict(X_new)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The `LinearRegression` class is based on the `scipy.linalg.lstsq()` function (the name stands for "least squares"), which you could call directly:
        """
    )
    return


@app.cell
def _(X_b, np, y):
    theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
    theta_best_svd
    return rank, residuals, s, theta_best_svd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This function computes $\mathbf{X}^+\mathbf{y}$, where $\mathbf{X}^{+}$ is the _pseudoinverse_ of $\mathbf{X}$ (specifically the Moore-Penrose inverse). You can use `np.linalg.pinv()` to compute the pseudoinverse directly:
        """
    )
    return


@app.cell
def _(X_b, np, y):
    np.linalg.pinv(X_b) @ y
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Gradient Descent
        ## Batch Gradient Descent
        """
    )
    return


@app.cell
def _(X_b, np, y):
    _eta = 0.1
    _n_epochs = 1000
    m_1 = len(X_b)
    np.random.seed(42)
    theta = np.random.randn(2, 1)
    for _epoch in range(_n_epochs):
        _gradients = 2 / m_1 * X_b.T @ (X_b @ theta - y)
        theta = theta - _eta * _gradients
    return m_1, theta


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The trained model parameters:
        """
    )
    return


@app.cell
def _(theta):
    theta
    return


@app.cell
def _(X, X_b, X_new, X_new_b, np, plt, save_fig, y):
    import matplotlib as mpl

    def plot_gradient_descent(theta, eta):
        m = len(X_b)
        plt.plot(X, y, 'b.')
        _n_epochs = 1000
        n_shown = 20
        theta_path = []
        for _epoch in range(_n_epochs):
            if _epoch < n_shown:
                y_predict = X_new_b @ theta
                color = mpl.colors.rgb2hex(plt.cm.OrRd(_epoch / n_shown + 0.15))
                plt.plot(X_new, y_predict, linestyle='solid', color=color)
            _gradients = 2 / m * X_b.T @ (X_b @ theta - y)
            theta = theta - _eta * _gradients
            theta_path.append(theta)
        plt.xlabel('$x_1$')
        plt.axis([0, 2, 0, 15])
        plt.grid()
        plt.title(f'$\\eta = {_eta}$')
        return theta_path
    np.random.seed(42)
    theta_1 = np.random.randn(2, 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(131)
    plot_gradient_descent(theta_1, eta=0.02)
    plt.ylabel('$y$', rotation=0)
    plt.subplot(132)
    theta_path_bgd = plot_gradient_descent(theta_1, eta=0.1)
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.subplot(133)
    plt.gca().axes.yaxis.set_ticklabels([])
    plot_gradient_descent(theta_1, eta=0.5)
    save_fig('gradient_descent_plot')
    plt.show()
    return mpl, plot_gradient_descent, theta_1, theta_path_bgd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Stochastic Gradient Descent
        """
    )
    return


@app.cell
def _():
    theta_path_sgd = []  # extra code – we need to store the path of theta in the
                         #              parameter space to plot the next figure
    return (theta_path_sgd,)


@app.cell
def _(X, X_b, X_new, X_new_b, m_1, mpl, np, plt, save_fig, theta_path_sgd, y):
    _n_epochs = 50
    _t0, _t1 = (5, 50)

    def _learning_schedule(t):
        return _t0 / (t + _t1)
    np.random.seed(42)
    theta_2 = np.random.randn(2, 1)
    n_shown = 20
    plt.figure(figsize=(6, 4))
    for _epoch in range(_n_epochs):
        for _iteration in range(m_1):
            if _epoch == 0 and _iteration < n_shown:
                y_predict_1 = X_new_b @ theta_2
                color = mpl.colors.rgb2hex(plt.cm.OrRd(_iteration / n_shown + 0.15))
                plt.plot(X_new, y_predict_1, color=color)
            random_index = np.random.randint(m_1)
            _xi = X_b[random_index:random_index + 1]
            _yi = y[random_index:random_index + 1]
            _gradients = 2 * _xi.T @ (_xi @ theta_2 - _yi)
            _eta = _learning_schedule(_epoch * m_1 + _iteration)
            theta_2 = theta_2 - _eta * _gradients
            theta_path_sgd.append(theta_2)
    plt.plot(X, y, 'b.')
    plt.xlabel('$x_1$')
    plt.ylabel('$y$', rotation=0)
    plt.axis([0, 2, 0, 15])
    plt.grid()
    save_fig('sgd_plot')
    plt.show()
    return color, n_shown, random_index, theta_2, y_predict_1


@app.cell
def _(theta_2):
    theta_2
    return


@app.cell
def _(X, y):
    from sklearn.linear_model import SGDRegressor

    sgd_reg = SGDRegressor(max_iter=1000, tol=1e-5, penalty=None, eta0=0.01,
                           n_iter_no_change=100, random_state=42)
    sgd_reg.fit(X, y.ravel())  # y.ravel() because fit() expects 1D targets
    return SGDRegressor, sgd_reg


@app.cell
def _(sgd_reg):
    sgd_reg.intercept_, sgd_reg.coef_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Mini-batch gradient descent
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The code in this section is used to generate the next figure, it is not in the book.
        """
    )
    return


@app.cell
def _(X_b, m_1, np, plt, save_fig, theta_path_bgd, theta_path_sgd, y):
    from math import ceil
    _n_epochs = 50
    minibatch_size = 20
    n_batches_per_epoch = ceil(m_1 / minibatch_size)
    np.random.seed(42)
    theta_3 = np.random.randn(2, 1)
    _t0, _t1 = (200, 1000)

    def _learning_schedule(t):
        return _t0 / (t + _t1)
    theta_path_mgd = []
    for _epoch in range(_n_epochs):
        shuffled_indices = np.random.permutation(m_1)
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for _iteration in range(0, n_batches_per_epoch):
            idx = _iteration * minibatch_size
            _xi = X_b_shuffled[idx:idx + minibatch_size]
            _yi = y_shuffled[idx:idx + minibatch_size]
            _gradients = 2 / minibatch_size * _xi.T @ (_xi @ theta_3 - _yi)
            _eta = _learning_schedule(_iteration)
            theta_3 = theta_3 - _eta * _gradients
            theta_path_mgd.append(theta_3)
    theta_path_bgd_1 = np.array(theta_path_bgd)
    theta_path_sgd_1 = np.array(theta_path_sgd)
    theta_path_mgd = np.array(theta_path_mgd)
    plt.figure(figsize=(7, 4))
    plt.plot(theta_path_sgd_1[:, 0], theta_path_sgd_1[:, 1], 'r-s', linewidth=1, label='Stochastic')
    plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], 'g-+', linewidth=2, label='Mini-batch')
    plt.plot(theta_path_bgd_1[:, 0], theta_path_bgd_1[:, 1], 'b-o', linewidth=3, label='Batch')
    plt.legend(loc='upper left')
    plt.xlabel('$\\theta_0$')
    plt.ylabel('$\\theta_1$   ', rotation=0)
    plt.axis([2.6, 4.6, 2.3, 3.4])
    plt.grid()
    save_fig('gradient_descent_paths_plot')
    plt.show()
    return (
        X_b_shuffled,
        ceil,
        idx,
        minibatch_size,
        n_batches_per_epoch,
        shuffled_indices,
        theta_3,
        theta_path_bgd_1,
        theta_path_mgd,
        theta_path_sgd_1,
        y_shuffled,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Polynomial Regression
        """
    )
    return


@app.cell
def _(np):
    np.random.seed(42)
    m_2 = 100
    X_1 = 6 * np.random.rand(m_2, 1) - 3
    y_1 = 0.5 * X_1 ** 2 + X_1 + 2 + np.random.randn(m_2, 1)
    return X_1, m_2, y_1


@app.cell
def _(X_1, plt, save_fig, y_1):
    plt.figure(figsize=(6, 4))
    plt.plot(X_1, y_1, 'b.')
    plt.xlabel('$x_1$')
    plt.ylabel('$y$', rotation=0)
    plt.axis([-3, 3, 0, 10])
    plt.grid()
    save_fig('quadratic_data_plot')
    plt.show()
    return


@app.cell
def _(X_1):
    from sklearn.preprocessing import PolynomialFeatures
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X_1)
    X_1[0]
    return PolynomialFeatures, X_poly, poly_features


@app.cell
def _(X_poly):
    X_poly[0]
    return


@app.cell
def _(LinearRegression, X_poly, y_1):
    lin_reg_1 = LinearRegression()
    lin_reg_1.fit(X_poly, y_1)
    (lin_reg_1.intercept_, lin_reg_1.coef_)
    return (lin_reg_1,)


@app.cell
def _(X_1, lin_reg_1, np, plt, poly_features, save_fig, y_1):
    X_new_1 = np.linspace(-3, 3, 100).reshape(100, 1)
    X_new_poly = poly_features.transform(X_new_1)
    y_new = lin_reg_1.predict(X_new_poly)
    plt.figure(figsize=(6, 4))
    plt.plot(X_1, y_1, 'b.')
    plt.plot(X_new_1, y_new, 'r-', linewidth=2, label='Predictions')
    plt.xlabel('$x_1$')
    plt.ylabel('$y$', rotation=0)
    plt.legend(loc='upper left')
    plt.axis([-3, 3, 0, 10])
    plt.grid()
    save_fig('quadratic_predictions_plot')
    plt.show()
    return X_new_1, X_new_poly, y_new


@app.cell
def _(LinearRegression, PolynomialFeatures, X_1, X_new_1, plt, save_fig, y_1):
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    plt.figure(figsize=(6, 4))
    for style, width, degree in (('r-+', 2, 1), ('b--', 2, 2), ('g-', 1, 300)):
        polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
        std_scaler = StandardScaler()
        lin_reg_2 = LinearRegression()
        _polynomial_regression = make_pipeline(polybig_features, std_scaler, lin_reg_2)
        _polynomial_regression.fit(X_1, y_1)
        y_newbig = _polynomial_regression.predict(X_new_1)
        label = f'{degree} degree{('s' if degree > 1 else '')}'
        plt.plot(X_new_1, y_newbig, style, label=label, linewidth=width)
    plt.plot(X_1, y_1, 'b.', linewidth=3)
    plt.legend(loc='upper left')
    plt.xlabel('$x_1$')
    plt.ylabel('$y$', rotation=0)
    plt.axis([-3, 3, 0, 10])
    plt.grid()
    save_fig('high_degree_polynomials_plot')
    plt.show()
    return (
        StandardScaler,
        degree,
        label,
        lin_reg_2,
        make_pipeline,
        polybig_features,
        std_scaler,
        style,
        width,
        y_newbig,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Learning Curves
        """
    )
    return


@app.cell
def _(LinearRegression, X_1, np, plt, save_fig, y_1):
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, valid_scores = learning_curve(LinearRegression(), X_1, y_1, train_sizes=np.linspace(0.01, 1.0, 40), cv=5, scoring='neg_root_mean_squared_error')
    _train_errors = -train_scores.mean(axis=1)
    _valid_errors = -valid_scores.mean(axis=1)
    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, _train_errors, 'r-+', linewidth=2, label='train')
    plt.plot(train_sizes, _valid_errors, 'b-', linewidth=3, label='valid')
    plt.xlabel('Training set size')
    plt.ylabel('RMSE')
    plt.grid()
    plt.legend(loc='upper right')
    plt.axis([0, 80, 0, 2.5])
    save_fig('underfitting_learning_curves_plot')
    plt.show()
    return learning_curve, train_scores, train_sizes, valid_scores


@app.cell
def _(
    LinearRegression,
    PolynomialFeatures,
    X_1,
    learning_curve,
    make_pipeline,
    np,
    y_1,
):
    _polynomial_regression = make_pipeline(PolynomialFeatures(degree=10, include_bias=False), LinearRegression())
    train_sizes_1, train_scores_1, valid_scores_1 = learning_curve(_polynomial_regression, X_1, y_1, train_sizes=np.linspace(0.01, 1.0, 40), cv=5, scoring='neg_root_mean_squared_error')
    return train_scores_1, train_sizes_1, valid_scores_1


@app.cell
def _(plt, save_fig, train_scores_1, train_sizes_1, valid_scores_1):
    _train_errors = -train_scores_1.mean(axis=1)
    _valid_errors = -valid_scores_1.mean(axis=1)
    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes_1, _train_errors, 'r-+', linewidth=2, label='train')
    plt.plot(train_sizes_1, _valid_errors, 'b-', linewidth=3, label='valid')
    plt.legend(loc='upper right')
    plt.xlabel('Training set size')
    plt.ylabel('RMSE')
    plt.grid()
    plt.axis([0, 80, 0, 2.5])
    save_fig('learning_curves_plot')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Regularized Linear Models
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Ridge Regression
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's generate a very small and noisy linear dataset:
        """
    )
    return


@app.cell
def _(np):
    np.random.seed(42)
    m_3 = 20
    X_2 = 3 * np.random.rand(m_3, 1)
    y_2 = 1 + 0.5 * X_2 + np.random.randn(m_3, 1) / 1.5
    X_new_2 = np.linspace(0, 3, 100).reshape(100, 1)
    return X_2, X_new_2, m_3, y_2


@app.cell
def _(X_2, plt, y_2):
    plt.figure(figsize=(6, 4))
    plt.plot(X_2, y_2, '.')
    plt.xlabel('$x_1$')
    plt.ylabel('$y$  ', rotation=0)
    plt.axis([0, 3, 0, 3.5])
    plt.grid()
    plt.show()
    return


@app.cell
def _(X_2, y_2):
    from sklearn.linear_model import Ridge
    ridge_reg = Ridge(alpha=0.1, solver='cholesky')
    ridge_reg.fit(X_2, y_2)
    ridge_reg.predict([[1.5]])
    return Ridge, ridge_reg


@app.cell
def _(
    LinearRegression,
    PolynomialFeatures,
    Ridge,
    StandardScaler,
    X_2,
    X_new_2,
    make_pipeline,
    plt,
    save_fig,
    y_2,
):
    def plot_model(model_class, polynomial, alphas, **model_kwargs):
        plt.plot(X_2, y_2, 'b.', linewidth=3)
        for _alpha, style in zip(alphas, ('b:', 'g--', 'r-')):
            if _alpha > 0:
                model = model_class(_alpha, **model_kwargs)
            else:
                model = LinearRegression()
            if polynomial:
                model = make_pipeline(PolynomialFeatures(degree=10, include_bias=False), StandardScaler(), model)
            model.fit(X_2, y_2)
            y_new_regul = model.predict(X_new_2)
            plt.plot(X_new_2, y_new_regul, style, linewidth=2, label=f'$\\alpha = {_alpha}$')
        plt.legend(loc='upper left')
        plt.xlabel('$x_1$')
        plt.axis([0, 3, 0, 3.5])
        plt.grid()
    plt.figure(figsize=(9, 3.5))
    plt.subplot(121)
    plot_model(Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
    plt.ylabel('$y$  ', rotation=0)
    plt.subplot(122)
    plot_model(Ridge, polynomial=True, alphas=(0, 10 ** (-5), 1), random_state=42)
    plt.gca().axes.yaxis.set_ticklabels([])
    save_fig('ridge_regression_plot')
    plt.show()
    return (plot_model,)


@app.cell
def _(SGDRegressor, X_2, m_3, y_2):
    sgd_reg_1 = SGDRegressor(penalty='l2', alpha=0.1 / m_3, tol=None, max_iter=1000, eta0=0.01, random_state=42)
    sgd_reg_1.fit(X_2, y_2.ravel())
    sgd_reg_1.predict([[1.5]])
    return (sgd_reg_1,)


@app.cell
def _(Ridge, X_2, y_2):
    ridge_reg_1 = Ridge(alpha=0.1, solver='sag', random_state=42)
    ridge_reg_1.fit(X_2, y_2)
    ridge_reg_1.predict([[1.5]])
    return (ridge_reg_1,)


@app.cell
def _(X_2, m_3, np, y_2):
    _alpha = 0.1
    A = np.array([[0.0, 0.0], [0.0, 1.0]])
    X_b_1 = np.c_[np.ones(m_3), X_2]
    np.linalg.inv(X_b_1.T @ X_b_1 + _alpha * A) @ X_b_1.T @ y_2
    return A, X_b_1


@app.cell
def _(ridge_reg_1):
    (ridge_reg_1.intercept_, ridge_reg_1.coef_)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Lasso Regression
        """
    )
    return


@app.cell
def _(X_2, y_2):
    from sklearn.linear_model import Lasso
    lasso_reg = Lasso(alpha=0.1)
    lasso_reg.fit(X_2, y_2)
    lasso_reg.predict([[1.5]])
    return Lasso, lasso_reg


@app.cell
def _(Lasso, plot_model, plt, save_fig):
    # extra code – this cell generates and saves Figure 4–18
    plt.figure(figsize=(9, 3.5))
    plt.subplot(121)
    plot_model(Lasso, polynomial=False, alphas=(0, 0.1, 1), random_state=42)
    plt.ylabel("$y$  ", rotation=0)
    plt.subplot(122)
    plot_model(Lasso, polynomial=True, alphas=(0, 1e-2, 1), random_state=42)
    plt.gca().axes.yaxis.set_ticklabels([])
    save_fig("lasso_regression_plot")
    plt.show()
    return


@app.cell
def _(np, plt, save_fig):
    t1a, t1b, t2a, t2b = (-1, 3, -1.5, 1.5)
    t1s = np.linspace(t1a, t1b, 500)
    t2s = np.linspace(t2a, t2b, 500)
    _t1, t2 = np.meshgrid(t1s, t2s)
    T = np.c_[_t1.ravel(), t2.ravel()]
    Xr = np.array([[1, 1], [1, -1], [1, 0.5]])
    yr = 2 * Xr[:, :1] + 0.5 * Xr[:, 1:]
    J = (1 / len(Xr) * ((T @ Xr.T - yr.T) ** 2).sum(axis=1)).reshape(_t1.shape)
    N1 = np.linalg.norm(T, ord=1, axis=1).reshape(_t1.shape)
    N2 = np.linalg.norm(T, ord=2, axis=1).reshape(_t1.shape)
    t_min_idx = np.unravel_index(J.argmin(), J.shape)
    t1_min, t2_min = (_t1[t_min_idx], t2[t_min_idx])
    t_init = np.array([[0.25], [-1]])

    def bgd_path(theta, X, y, l1, l2, core=1, eta=0.05, n_iterations=200):
        path = [theta]
        for _iteration in range(n_iterations):
            _gradients = core * 2 / len(X) * X.T @ (X @ theta - y) + l1 * np.sign(theta) + l2 * theta
            theta = theta - _eta * _gradients
            path.append(theta)
        return np.array(path)
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10.1, 8))
    for i, N, l1, l2, title in ((0, N1, 2.0, 0, 'Lasso'), (1, N2, 0, 2.0, 'Ridge')):
        JR = J + l1 * N1 + l2 * 0.5 * N2 ** 2
        tr_min_idx = np.unravel_index(JR.argmin(), JR.shape)
        t1r_min, t2r_min = (_t1[tr_min_idx], t2[tr_min_idx])
        levels = np.exp(np.linspace(0, 1, 20)) - 1
        levelsJ = levels * (J.max() - J.min()) + J.min()
        levelsJR = levels * (JR.max() - JR.min()) + JR.min()
        levelsN = np.linspace(0, N.max(), 10)
        path_J = bgd_path(t_init, Xr, yr, l1=0, l2=0)
        path_JR = bgd_path(t_init, Xr, yr, l1, l2)
        path_N = bgd_path(theta=np.array([[2.0], [0.5]]), X=Xr, y=yr, l1=np.sign(l1) / 3, l2=np.sign(l2), core=0)
        ax = axes[i, 0]
        ax.grid()
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.contourf(_t1, t2, N / 2.0, levels=levelsN)
        ax.plot(path_N[:, 0], path_N[:, 1], 'y--')
        ax.plot(0, 0, 'ys')
        ax.plot(t1_min, t2_min, 'ys')
        ax.set_title(f'$\\ell_{i + 1}$ penalty')
        ax.axis([t1a, t1b, t2a, t2b])
        if i == 1:
            ax.set_xlabel('$\\theta_1$')
        ax.set_ylabel('$\\theta_2$', rotation=0)
        ax = axes[i, 1]
        ax.grid()
        ax.axhline(y=0, color='k')
        ax.axvline(x=0, color='k')
        ax.contourf(_t1, t2, JR, levels=levelsJR, alpha=0.9)
        ax.plot(path_JR[:, 0], path_JR[:, 1], 'w-o')
        ax.plot(path_N[:, 0], path_N[:, 1], 'y--')
        ax.plot(0, 0, 'ys')
        ax.plot(t1_min, t2_min, 'ys')
        ax.plot(t1r_min, t2r_min, 'rs')
        ax.set_title(title)
        ax.axis([t1a, t1b, t2a, t2b])
        if i == 1:
            ax.set_xlabel('$\\theta_1$')
    save_fig('lasso_vs_ridge_plot')
    plt.show()
    return (
        J,
        JR,
        N,
        N1,
        N2,
        T,
        Xr,
        ax,
        axes,
        bgd_path,
        fig,
        i,
        l1,
        l2,
        levels,
        levelsJ,
        levelsJR,
        levelsN,
        path_J,
        path_JR,
        path_N,
        t1_min,
        t1a,
        t1b,
        t1r_min,
        t1s,
        t2,
        t2_min,
        t2a,
        t2b,
        t2r_min,
        t2s,
        t_init,
        t_min_idx,
        title,
        tr_min_idx,
        yr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Elastic Net
        """
    )
    return


@app.cell
def _(X_2, y_2):
    from sklearn.linear_model import ElasticNet
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elastic_net.fit(X_2, y_2)
    elastic_net.predict([[1.5]])
    return ElasticNet, elastic_net


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Early Stopping
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's go back to the quadratic dataset we used earlier:
        """
    )
    return


@app.cell
def _(
    PolynomialFeatures,
    SGDRegressor,
    StandardScaler,
    make_pipeline,
    np,
    plt,
    save_fig,
):
    from copy import deepcopy
    from sklearn.metrics import mean_squared_error
    np.random.seed(42)
    m_4 = 100
    X_3 = 6 * np.random.rand(m_4, 1) - 3
    y_3 = 0.5 * X_3 ** 2 + X_3 + 2 + np.random.randn(m_4, 1)
    X_train, y_train = (X_3[:m_4 // 2], y_3[:m_4 // 2, 0])
    X_valid, y_valid = (X_3[m_4 // 2:], y_3[m_4 // 2:, 0])
    preprocessing = make_pipeline(PolynomialFeatures(degree=90, include_bias=False), StandardScaler())
    X_train_prep = preprocessing.fit_transform(X_train)
    X_valid_prep = preprocessing.transform(X_valid)
    sgd_reg_2 = SGDRegressor(penalty=None, eta0=0.002, random_state=42)
    _n_epochs = 500
    best_valid_rmse = float('inf')
    _train_errors, val_errors = ([], [])
    for _epoch in range(_n_epochs):
        sgd_reg_2.partial_fit(X_train_prep, y_train)
        y_valid_predict = sgd_reg_2.predict(X_valid_prep)
        val_error = mean_squared_error(y_valid, y_valid_predict, squared=False)
        if val_error < best_valid_rmse:
            best_valid_rmse = val_error
            best_model = deepcopy(sgd_reg_2)
        y_train_predict = sgd_reg_2.predict(X_train_prep)
        train_error = mean_squared_error(y_train, y_train_predict, squared=False)
        val_errors.append(val_error)
        _train_errors.append(train_error)
    best_epoch = np.argmin(val_errors)
    plt.figure(figsize=(6, 4))
    plt.annotate('Best model', xy=(best_epoch, best_valid_rmse), xytext=(best_epoch, best_valid_rmse + 0.5), ha='center', arrowprops=dict(facecolor='black', shrink=0.05))
    plt.plot([0, _n_epochs], [best_valid_rmse, best_valid_rmse], 'k:', linewidth=2)
    plt.plot(val_errors, 'b-', linewidth=3, label='Validation set')
    plt.plot(best_epoch, best_valid_rmse, 'bo')
    plt.plot(_train_errors, 'r--', linewidth=2, label='Training set')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.axis([0, _n_epochs, 0, 3.5])
    plt.grid()
    save_fig('early_stopping_plot')
    plt.show()
    return (
        X_3,
        X_train,
        X_train_prep,
        X_valid,
        X_valid_prep,
        best_epoch,
        best_model,
        best_valid_rmse,
        deepcopy,
        m_4,
        mean_squared_error,
        preprocessing,
        sgd_reg_2,
        train_error,
        val_error,
        val_errors,
        y_3,
        y_train,
        y_train_predict,
        y_valid,
        y_valid_predict,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Logistic Regression
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Estimating Probabilities
        """
    )
    return


@app.cell
def _(np, plt, save_fig):
    # extra code – generates and saves Figure 4–21

    lim = 6
    t = np.linspace(-lim, lim, 100)
    sig = 1 / (1 + np.exp(-t))

    plt.figure(figsize=(8, 3))
    plt.plot([-lim, lim], [0, 0], "k-")
    plt.plot([-lim, lim], [0.5, 0.5], "k:")
    plt.plot([-lim, lim], [1, 1], "k:")
    plt.plot([0, 0], [-1.1, 1.1], "k-")
    plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \dfrac{1}{1 + e^{-t}}$")
    plt.xlabel("t")
    plt.legend(loc="upper left")
    plt.axis([-lim, lim, -0.1, 1.1])
    plt.gca().set_yticks([0, 0.25, 0.5, 0.75, 1])
    plt.grid()
    save_fig("logistic_function_plot")
    plt.show()
    return lim, sig, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Decision Boundaries
        """
    )
    return


@app.cell
def _():
    from sklearn.datasets import load_iris

    iris = load_iris(as_frame=True)
    list(iris)
    return iris, load_iris


@app.cell
def _(iris):
    print(iris.DESCR)  # extra code – it's a bit too long
    return


@app.cell
def _(iris):
    iris.data.head(3)
    return


@app.cell
def _(iris):
    iris.target.head(3)  # note that the instances are not shuffled
    return


@app.cell
def _(iris):
    iris.target_names
    return


@app.cell
def _(iris):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    X_4 = iris.data[['petal width (cm)']].values
    y_4 = iris.target_names[iris.target] == 'virginica'
    X_train_1, X_test, y_train_1, y_test = train_test_split(X_4, y_4, random_state=42)
    log_reg = LogisticRegression(random_state=42)
    log_reg.fit(X_train_1, y_train_1)
    return (
        LogisticRegression,
        X_4,
        X_test,
        X_train_1,
        log_reg,
        train_test_split,
        y_4,
        y_test,
        y_train_1,
    )


@app.cell
def _(X_train_1, log_reg, np, plt, save_fig, y_train_1):
    X_new_3 = np.linspace(0, 3, 1000).reshape(-1, 1)
    _y_proba = log_reg.predict_proba(X_new_3)
    decision_boundary = X_new_3[_y_proba[:, 1] >= 0.5][0, 0]
    plt.figure(figsize=(8, 3))
    plt.plot(X_new_3, _y_proba[:, 0], 'b--', linewidth=2, label='Not Iris virginica proba')
    plt.plot(X_new_3, _y_proba[:, 1], 'g-', linewidth=2, label='Iris virginica proba')
    plt.plot([decision_boundary, decision_boundary], [0, 1], 'k:', linewidth=2, label='Decision boundary')
    plt.arrow(x=decision_boundary, y=0.08, dx=-0.3, dy=0, head_width=0.05, head_length=0.1, fc='b', ec='b')
    plt.arrow(x=decision_boundary, y=0.92, dx=0.3, dy=0, head_width=0.05, head_length=0.1, fc='g', ec='g')
    plt.plot(X_train_1[y_train_1 == 0], y_train_1[y_train_1 == 0], 'bs')
    plt.plot(X_train_1[y_train_1 == 1], y_train_1[y_train_1 == 1], 'g^')
    plt.xlabel('Petal width (cm)')
    plt.ylabel('Probability')
    plt.legend(loc='center left')
    plt.axis([0, 3, -0.02, 1.02])
    plt.grid()
    save_fig('logistic_regression_plot')
    plt.show()
    return X_new_3, decision_boundary


@app.cell
def _(decision_boundary):
    decision_boundary
    return


@app.cell
def _(log_reg):
    log_reg.predict([[1.7], [1.5]])
    return


@app.cell
def _(LogisticRegression, iris, np, plt, save_fig, train_test_split):
    X_5 = iris.data[['petal length (cm)', 'petal width (cm)']].values
    y_5 = iris.target_names[iris.target] == 'virginica'
    X_train_2, X_test_1, y_train_2, y_test_1 = train_test_split(X_5, y_5, random_state=42)
    log_reg_1 = LogisticRegression(C=2, random_state=42)
    log_reg_1.fit(X_train_2, y_train_2)
    _x0, _x1 = np.meshgrid(np.linspace(2.9, 7, 500).reshape(-1, 1), np.linspace(0.8, 2.7, 200).reshape(-1, 1))
    X_new_4 = np.c_[_x0.ravel(), _x1.ravel()]
    _y_proba = log_reg_1.predict_proba(X_new_4)
    _zz = _y_proba[:, 1].reshape(_x0.shape)
    left_right = np.array([2.9, 7])
    boundary = -((log_reg_1.coef_[0, 0] * left_right + log_reg_1.intercept_[0]) / log_reg_1.coef_[0, 1])
    plt.figure(figsize=(10, 4))
    plt.plot(X_train_2[y_train_2 == 0, 0], X_train_2[y_train_2 == 0, 1], 'bs')
    plt.plot(X_train_2[y_train_2 == 1, 0], X_train_2[y_train_2 == 1, 1], 'g^')
    _contour = plt.contour(_x0, _x1, _zz, cmap=plt.cm.brg)
    plt.clabel(_contour, inline=1)
    plt.plot(left_right, boundary, 'k--', linewidth=3)
    plt.text(3.5, 1.27, 'Not Iris virginica', color='b', ha='center')
    plt.text(6.5, 2.3, 'Iris virginica', color='g', ha='center')
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.axis([2.9, 7, 0.8, 2.7])
    plt.grid()
    save_fig('logistic_regression_contour_plot')
    plt.show()
    return (
        X_5,
        X_new_4,
        X_test_1,
        X_train_2,
        boundary,
        left_right,
        log_reg_1,
        y_5,
        y_test_1,
        y_train_2,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Softmax Regression
        """
    )
    return


@app.cell
def _(LogisticRegression, iris, train_test_split):
    X_6 = iris.data[['petal length (cm)', 'petal width (cm)']].values
    y_6 = iris['target']
    X_train_3, X_test_2, y_train_3, y_test_2 = train_test_split(X_6, y_6, random_state=42)
    softmax_reg = LogisticRegression(C=30, random_state=42)
    softmax_reg.fit(X_train_3, y_train_3)
    return X_6, X_test_2, X_train_3, softmax_reg, y_6, y_test_2, y_train_3


@app.cell
def _(softmax_reg):
    softmax_reg.predict([[5, 2]])
    return


@app.cell
def _(softmax_reg):
    softmax_reg.predict_proba([[5, 2]]).round(2)
    return


@app.cell
def _(X_6, np, plt, save_fig, softmax_reg, y_6):
    from matplotlib.colors import ListedColormap
    _custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    _x0, _x1 = np.meshgrid(np.linspace(0, 8, 500).reshape(-1, 1), np.linspace(0, 3.5, 200).reshape(-1, 1))
    X_new_5 = np.c_[_x0.ravel(), _x1.ravel()]
    _y_proba = softmax_reg.predict_proba(X_new_5)
    y_predict_2 = softmax_reg.predict(X_new_5)
    _zz1 = _y_proba[:, 1].reshape(_x0.shape)
    _zz = y_predict_2.reshape(_x0.shape)
    plt.figure(figsize=(10, 4))
    plt.plot(X_6[y_6 == 2, 0], X_6[y_6 == 2, 1], 'g^', label='Iris virginica')
    plt.plot(X_6[y_6 == 1, 0], X_6[y_6 == 1, 1], 'bs', label='Iris versicolor')
    plt.plot(X_6[y_6 == 0, 0], X_6[y_6 == 0, 1], 'yo', label='Iris setosa')
    plt.contourf(_x0, _x1, _zz, cmap=_custom_cmap)
    _contour = plt.contour(_x0, _x1, _zz1, cmap='hot')
    plt.clabel(_contour, inline=1)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.legend(loc='center left')
    plt.axis([0.5, 7, 0, 3.5])
    plt.grid()
    save_fig('softmax_regression_contour_plot')
    plt.show()
    return ListedColormap, X_new_5, y_predict_2


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
        ## 1. to 11.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        1. If you have a training set with millions of features you can use Stochastic Gradient Descent or Mini-batch Gradient Descent, and perhaps Batch Gradient Descent if the training set fits in memory. But you cannot use the Normal Equation or the SVD approach because the computational complexity grows quickly (more than quadratically) with the number of features.
        2. If the features in your training set have very different scales, the cost function will have the shape of an elongated bowl, so the Gradient Descent algorithms will take a long time to converge. To solve this you should scale the data before training the model. Note that the Normal Equation or SVD approach will work just fine without scaling. Moreover, regularized models may converge to a suboptimal solution if the features are not scaled: since regularization penalizes large weights, features with smaller values will tend to be ignored compared to features with larger values.
        3. Gradient Descent cannot get stuck in a local minimum when training a Logistic Regression model because the cost function is convex. _Convex_ means that if you draw a straight line between any two points on the curve, the line never crosses the curve.
        4. If the optimization problem is convex (such as Linear Regression or Logistic Regression), and assuming the learning rate is not too high, then all Gradient Descent algorithms will approach the global optimum and end up producing fairly similar models. However, unless you gradually reduce the learning rate, Stochastic GD and Mini-batch GD will never truly converge; instead, they will keep jumping back and forth around the global optimum. This means that even if you let them run for a very long time, these Gradient Descent algorithms will produce slightly different models.
        5. If the validation error consistently goes up after every epoch, then one possibility is that the learning rate is too high and the algorithm is diverging. If the training error also goes up, then this is clearly the problem and you should reduce the learning rate. However, if the training error is not going up, then your model is overfitting the training set and you should stop training.
        6. Due to their random nature, neither Stochastic Gradient Descent nor Mini-batch Gradient Descent is guaranteed to make progress at every single training iteration. So if you immediately stop training when the validation error goes up, you may stop much too early, before the optimum is reached. A better option is to save the model at regular intervals; then, when it has not improved for a long time (meaning it will probably never beat the record), you can revert to the best saved model.
        7. Stochastic Gradient Descent has the fastest training iteration since it considers only one training instance at a time, so it is generally the first to reach the vicinity of the global optimum (or Mini-batch GD with a very small mini-batch size). However, only Batch Gradient Descent will actually converge, given enough training time. As mentioned, Stochastic GD and Mini-batch GD will bounce around the optimum, unless you gradually reduce the learning rate.
        8. If the validation error is much higher than the training error, this is likely because your model is overfitting the training set. One way to try to fix this is to reduce the polynomial degree: a model with fewer degrees of freedom is less likely to overfit. Another thing you can try is to regularize the model—for example, by adding an ℓ₂ penalty (Ridge) or an ℓ₁ penalty (Lasso) to the cost function. This will also reduce the degrees of freedom of the model. Lastly, you can try to increase the size of the training set.
        9. If both the training error and the validation error are almost equal and fairly high, the model is likely underfitting the training set, which means it has a high bias. You should try reducing the regularization hyperparameter _α_.
        10. Let's see:
          * A model with some regularization typically performs better than a model without any regularization, so you should generally prefer Ridge Regression over plain Linear Regression.
          * Lasso Regression uses an ℓ₁ penalty, which tends to push the weights down to exactly zero. This leads to sparse models, where all weights are zero except for the most important weights. This is a way to perform feature selection automatically, which is good if you suspect that only a few features actually matter. When you are not sure, you should prefer Ridge Regression.
          * Elastic Net is generally preferred over Lasso since Lasso may behave erratically in some cases (when several features are strongly correlated or when there are more features than training instances). However, it does add an extra hyperparameter to tune. If you want Lasso without the erratic behavior, you can just use Elastic Net with an `l1_ratio` close to 1.
        11. If you want to classify pictures as outdoor/indoor and daytime/nighttime, since these are not exclusive classes (i.e., all four combinations are possible) you should train two Logistic Regression classifiers.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 12. Batch Gradient Descent with early stopping for Softmax Regression
        Exercise: _Implement Batch Gradient Descent with early stopping for Softmax Regression without using Scikit-Learn, only NumPy. Use it on a classification task such as the iris dataset._
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's start by loading the data. We will just reuse the Iris dataset we loaded earlier.
        """
    )
    return


@app.cell
def _(iris):
    X_7 = iris.data[['petal length (cm)', 'petal width (cm)']].values
    y_7 = iris['target'].values
    return X_7, y_7


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We need to add the bias term for every instance ($x_0 = 1$). The easiest option to do this would be to use Scikit-Learn's `add_dummy_feature()` function, but the point of this exercise is to get a better understanding of the algorithms by implementing them manually. So here is one possible implementation:
        """
    )
    return


@app.cell
def _(X_7, np):
    X_with_bias = np.c_[np.ones(len(X_7)), X_7]
    return (X_with_bias,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The easiest option to split the dataset into a training set, a validation set and a test set would be to use Scikit-Learn's `train_test_split()` function, but again, we want to do it manually:
        """
    )
    return


@app.cell
def _(X_with_bias, np, y_7):
    test_ratio = 0.2
    validation_ratio = 0.2
    total_size = len(X_with_bias)
    test_size = int(total_size * test_ratio)
    validation_size = int(total_size * validation_ratio)
    train_size = total_size - test_size - validation_size
    np.random.seed(42)
    rnd_indices = np.random.permutation(total_size)
    X_train_4 = X_with_bias[rnd_indices[:train_size]]
    y_train_4 = y_7[rnd_indices[:train_size]]
    X_valid_1 = X_with_bias[rnd_indices[train_size:-test_size]]
    y_valid_1 = y_7[rnd_indices[train_size:-test_size]]
    X_test_3 = X_with_bias[rnd_indices[-test_size:]]
    y_test_3 = y_7[rnd_indices[-test_size:]]
    return (
        X_test_3,
        X_train_4,
        X_valid_1,
        rnd_indices,
        test_ratio,
        test_size,
        total_size,
        train_size,
        validation_ratio,
        validation_size,
        y_test_3,
        y_train_4,
        y_valid_1,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The targets are currently class indices (0, 1 or 2), but we need target class probabilities to train the Softmax Regression model. Each instance will have target class probabilities equal to 0.0 for all classes except for the target class which will have a probability of 1.0 (in other words, the vector of class probabilities for any given instance is a one-hot vector). Let's write a small function to convert the vector of class indices into a matrix containing a one-hot vector for each instance. To understand this code, you need to know that `np.diag(np.ones(n))` creates an n×n matrix full of 0s except for 1s on the main diagonal. Moreover, if `a` is a NumPy array, then `a[[1, 3, 2]]` returns an array with 3 rows equal to `a[1]`, `a[3]` and `a[2]` (this is [advanced NumPy indexing](https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing)).
        """
    )
    return


@app.cell
def _(np):
    def to_one_hot(y):
        return np.diag(np.ones(y.max() + 1))[y]
    return (to_one_hot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's test this function on the first 10 instances:
        """
    )
    return


@app.cell
def _(y_train_4):
    y_train_4[:10]
    return


@app.cell
def _(to_one_hot, y_train_4):
    to_one_hot(y_train_4[:10])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Looks good, so let's create the target class probabilities matrix for the training set and the test set:
        """
    )
    return


@app.cell
def _(to_one_hot, y_test_3, y_train_4, y_valid_1):
    Y_train_one_hot = to_one_hot(y_train_4)
    Y_valid_one_hot = to_one_hot(y_valid_1)
    Y_test_one_hot = to_one_hot(y_test_3)
    return Y_test_one_hot, Y_train_one_hot, Y_valid_one_hot


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's scale the inputs. We compute the mean and standard deviation of each feature on the training set (except for the bias feature), then we center and scale each feature in the training set, the validation set, and the test set:
        """
    )
    return


@app.cell
def _(X_test_3, X_train_4, X_valid_1):
    mean = X_train_4[:, 1:].mean(axis=0)
    std = X_train_4[:, 1:].std(axis=0)
    X_train_4[:, 1:] = (X_train_4[:, 1:] - mean) / std
    X_valid_1[:, 1:] = (X_valid_1[:, 1:] - mean) / std
    X_test_3[:, 1:] = (X_test_3[:, 1:] - mean) / std
    return mean, std


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's implement the Softmax function. Recall that it is defined by the following equation:

        $\sigma\left(\mathbf{s}(\mathbf{x})\right)_k = \dfrac{\exp\left(s_k(\mathbf{x})\right)}{\sum\limits_{j=1}^{K}{\exp\left(s_j(\mathbf{x})\right)}}$
        """
    )
    return


@app.cell
def _(np):
    def softmax(logits):
        exps = np.exp(_logits)
        exp_sums = exps.sum(axis=1, keepdims=True)
        return exps / exp_sums
    return (softmax,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We are almost ready to start training. Let's define the number of inputs and outputs:
        """
    )
    return


@app.cell
def _(X_train_4, np, y_train_4):
    n_inputs = X_train_4.shape[1]
    n_outputs = len(np.unique(y_train_4))
    return n_inputs, n_outputs


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now here comes the hardest part: training! Theoretically, it's simple: it's just a matter of translating the math equations into Python code. But in practice, it can be quite tricky: in particular, it's easy to mix up the order of the terms, or the indices. You can even end up with code that looks like it's working but is actually not computing exactly the right thing. When unsure, you should write down the shape of each term in the equation and make sure the corresponding terms in your code match closely. It can also help to evaluate each term independently and print them out. The good news it that you won't have to do this everyday, since all this is well implemented by Scikit-Learn, but it will help you understand what's going on under the hood.

        So the equations we will need are the cost function:

        $J(\mathbf{\Theta}) =
        - \dfrac{1}{m}\sum\limits_{i=1}^{m}\sum\limits_{k=1}^{K}{y_k^{(i)}\log\left(\hat{p}_k^{(i)}\right)}$

        And the equation for the gradients:

        $\nabla_{\mathbf{\theta}^{(k)}} \, J(\mathbf{\Theta}) = \dfrac{1}{m} \sum\limits_{i=1}^{m}{ \left ( \hat{p}^{(i)}_k - y_k^{(i)} \right ) \mathbf{x}^{(i)}}$

        Note that $\log\left(\hat{p}_k^{(i)}\right)$ may not be computable if $\hat{p}_k^{(i)} = 0$. So we will add a tiny value $\epsilon$ to $\log\left(\hat{p}_k^{(i)}\right)$ to avoid getting `nan` values.
        """
    )
    return


@app.cell
def _(
    X_train_4,
    X_valid_1,
    Y_train_one_hot,
    Y_valid_one_hot,
    n_inputs,
    n_outputs,
    np,
    softmax,
):
    _eta = 0.5
    _n_epochs = 5001
    m_5 = len(X_train_4)
    _epsilon = 1e-05
    np.random.seed(42)
    Theta = np.random.randn(n_inputs, n_outputs)
    for _epoch in range(_n_epochs):
        _logits = X_train_4 @ Theta
        _Y_proba = softmax(_logits)
        if _epoch % 1000 == 0:
            _Y_proba_valid = softmax(X_valid_1 @ Theta)
            _xentropy_losses = -(Y_valid_one_hot * np.log(_Y_proba_valid + _epsilon))
            print(_epoch, _xentropy_losses.sum(axis=1).mean())
        _error = _Y_proba - Y_train_one_hot
        _gradients = 1 / m_5 * X_train_4.T @ _error
        Theta = Theta - _eta * _gradients
    return Theta, m_5


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And that's it! The Softmax model is trained. Let's look at the model parameters:
        """
    )
    return


@app.cell
def _(Theta):
    Theta
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's make predictions for the validation set and check the accuracy score:
        """
    )
    return


@app.cell
def _(Theta, X_valid_1, softmax, y_valid_1):
    _logits = X_valid_1 @ Theta
    _Y_proba = softmax(_logits)
    y_predict_3 = _Y_proba.argmax(axis=1)
    _accuracy_score = (y_predict_3 == y_valid_1).mean()
    _accuracy_score
    return (y_predict_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Well, this model looks pretty ok. For the sake of the exercise, let's add a bit of $\ell_2$ regularization. The following training code is similar to the one above, but the loss now has an additional $\ell_2$ penalty, and the gradients have the proper additional term (note that we don't regularize the first element of `Theta` since this corresponds to the bias term). Also, let's try increasing the learning rate `eta`.
        """
    )
    return


@app.cell
def _(
    X_train_4,
    X_valid_1,
    Y_train_one_hot,
    Y_valid_one_hot,
    n_inputs,
    n_outputs,
    np,
    softmax,
):
    _eta = 0.5
    _n_epochs = 5001
    m_6 = len(X_train_4)
    _epsilon = 1e-05
    _alpha = 0.01
    np.random.seed(42)
    Theta_1 = np.random.randn(n_inputs, n_outputs)
    for _epoch in range(_n_epochs):
        _logits = X_train_4 @ Theta_1
        _Y_proba = softmax(_logits)
        if _epoch % 1000 == 0:
            _Y_proba_valid = softmax(X_valid_1 @ Theta_1)
            _xentropy_losses = -(Y_valid_one_hot * np.log(_Y_proba_valid + _epsilon))
            _l2_loss = 1 / 2 * (Theta_1[1:] ** 2).sum()
            _total_loss = _xentropy_losses.sum(axis=1).mean() + _alpha * _l2_loss
            print(_epoch, _total_loss.round(4))
        _error = _Y_proba - Y_train_one_hot
        _gradients = 1 / m_6 * X_train_4.T @ _error
        _gradients = _gradients + np.r_[np.zeros([1, n_outputs]), _alpha * Theta_1[1:]]
        Theta_1 = Theta_1 - _eta * _gradients
    return Theta_1, m_6


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Because of the additional $\ell_2$ penalty, the loss seems greater than earlier, but perhaps this model will perform better? Let's find out:
        """
    )
    return


@app.cell
def _(Theta_1, X_valid_1, softmax, y_valid_1):
    _logits = X_valid_1 @ Theta_1
    _Y_proba = softmax(_logits)
    y_predict_4 = _Y_proba.argmax(axis=1)
    _accuracy_score = (y_predict_4 == y_valid_1).mean()
    _accuracy_score
    return (y_predict_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this case, the $\ell_2$ penalty did not change the test accuracy. Perhaps try fine-tuning `alpha`?
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's add early stopping. For this we just need to measure the loss on the validation set at every iteration and stop when the error starts growing.
        """
    )
    return


@app.cell
def _(
    X_train_4,
    X_valid_1,
    Y_train_one_hot,
    Y_valid_one_hot,
    n_inputs,
    n_outputs,
    np,
    softmax,
):
    _eta = 0.5
    _n_epochs = 50001
    m_7 = len(X_train_4)
    _epsilon = 1e-05
    C = 100
    best_loss = np.infty
    np.random.seed(42)
    Theta_2 = np.random.randn(n_inputs, n_outputs)
    for _epoch in range(_n_epochs):
        _logits = X_train_4 @ Theta_2
        _Y_proba = softmax(_logits)
        _Y_proba_valid = softmax(X_valid_1 @ Theta_2)
        _xentropy_losses = -(Y_valid_one_hot * np.log(_Y_proba_valid + _epsilon))
        _l2_loss = 1 / 2 * (Theta_2[1:] ** 2).sum()
        _total_loss = _xentropy_losses.sum(axis=1).mean() + 1 / C * _l2_loss
        if _epoch % 1000 == 0:
            print(_epoch, _total_loss.round(4))
        if _total_loss < best_loss:
            best_loss = _total_loss
        else:
            print(_epoch - 1, best_loss.round(4))
            print(_epoch, _total_loss.round(4), 'early stopping!')
            break
        _error = _Y_proba - Y_train_one_hot
        _gradients = 1 / m_7 * X_train_4.T @ _error
        _gradients = _gradients + np.r_[np.zeros([1, n_outputs]), 1 / C * Theta_2[1:]]
        Theta_2 = Theta_2 - _eta * _gradients
    return C, Theta_2, best_loss, m_7


@app.cell
def _(Theta_2, X_valid_1, softmax, y_valid_1):
    _logits = X_valid_1 @ Theta_2
    _Y_proba = softmax(_logits)
    y_predict_5 = _Y_proba.argmax(axis=1)
    _accuracy_score = (y_predict_5 == y_valid_1).mean()
    _accuracy_score
    return (y_predict_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Oh well, still no change in validation accuracy, but at least early stopping shortened training a bit.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's plot the model's predictions on the whole dataset (remember to scale all features fed to the model):
        """
    )
    return


@app.cell
def _(Theta_2, X_7, mean, mpl, np, plt, softmax, std, y_7):
    _custom_cmap = mpl.colors.ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    _x0, _x1 = np.meshgrid(np.linspace(0, 8, 500).reshape(-1, 1), np.linspace(0, 3.5, 200).reshape(-1, 1))
    X_new_6 = np.c_[_x0.ravel(), _x1.ravel()]
    X_new_6 = (X_new_6 - mean) / std
    X_new_with_bias = np.c_[np.ones(len(X_new_6)), X_new_6]
    _logits = X_new_with_bias @ Theta_2
    _Y_proba = softmax(_logits)
    y_predict_6 = _Y_proba.argmax(axis=1)
    _zz1 = _Y_proba[:, 1].reshape(_x0.shape)
    _zz = y_predict_6.reshape(_x0.shape)
    plt.figure(figsize=(10, 4))
    plt.plot(X_7[y_7 == 2, 0], X_7[y_7 == 2, 1], 'g^', label='Iris virginica')
    plt.plot(X_7[y_7 == 1, 0], X_7[y_7 == 1, 1], 'bs', label='Iris versicolor')
    plt.plot(X_7[y_7 == 0, 0], X_7[y_7 == 0, 1], 'yo', label='Iris setosa')
    plt.contourf(_x0, _x1, _zz, cmap=_custom_cmap)
    _contour = plt.contour(_x0, _x1, _zz1, cmap='hot')
    plt.clabel(_contour, inline=1)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.legend(loc='upper left')
    plt.axis([0, 7, 0, 3.5])
    plt.grid()
    plt.show()
    return X_new_6, X_new_with_bias, y_predict_6


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And now let's measure the final model's accuracy on the test set:
        """
    )
    return


@app.cell
def _(Theta_2, X_test_3, softmax, y_test_3):
    _logits = X_test_3 @ Theta_2
    _Y_proba = softmax(_logits)
    y_predict_7 = _Y_proba.argmax(axis=1)
    _accuracy_score = (y_predict_7 == y_test_3).mean()
    _accuracy_score
    return (y_predict_7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Well we get even better performance on the test set. This variability is likely due to the very small size of the dataset: depending on how you sample the training set, validation set and the test set, you can get quite different results. Try changing the random seed and running the code again a few times, you will see that the results will vary.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

