import marimo

__generated_with = "0.11.20"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Chapter 1 – The Machine Learning landscape**

        _This notebook contains the code examples in chapter 1. You'll also find the exercise solutions at the end of the notebook. The rest of this notebook is used to generate `lifesat.csv` from the original data sources, and some of this chapter's figures._

        You're welcome to go through the code in this notebook if you want, but the real action starts in the next chapter.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Code example 1-1""")
    return


@app.cell
def _():
    import duckdb

    # Download and prepare the data
    lifesat_url = "https://raw.gitmirror.com/ageron/data/main/lifesat/lifesat.csv"
    lifesat = duckdb.read_csv(lifesat_url)
    lifesat
    return duckdb, lifesat, lifesat_url


@app.cell
def _(lifesat):
    # Visualize the data
    lifesat.pl().plot.scatter(
        x="GDP per capita (USD)", 
        y="Life satisfaction"
    ).properties(
        title="Life satisfaction vs. GDP per capita (USD)",
        width=400,
        height=200,
    )
    return


@app.cell
def _(lifesat):
    from sklearn.linear_model import LinearRegression
    _model = LinearRegression()
    X = lifesat['GDP per capita (USD)'].pl()
    y = lifesat['Life satisfaction'].pl()
    _model.fit(X, y)
    X_new = [[37655.2]]
    _model.predict(X_new)
    return LinearRegression, X, X_new, y


@app.cell
def _(X, X_new, y):
    from sklearn.neighbors import KNeighborsRegressor
    _model = KNeighborsRegressor(n_neighbors=3)
    _model.fit(X, y)
    _model.predict(X_new)
    return (KNeighborsRegressor,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Exercise Solutions""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        1. Machine Learning is about building systems that can learn from data. Learning means getting better at some task, given some performance measure.
        2. Machine Learning is great for complex problems for which we have no algorithmic solution, to replace long lists of hand-tuned rules, to build systems that adapt to fluctuating environments, and finally to help humans learn (e.g., data mining).
        3. A labeled training set is a training set that contains the desired solution (a.k.a. a label) for each instance.
        4. The two most common supervised tasks are regression and classification.
        5. Common unsupervised tasks include clustering, visualization, dimensionality reduction, and association rule learning.
        6. Reinforcement Learning is likely to perform best if we want a robot to learn to walk in various unknown terrains, since this is typically the type of problem that Reinforcement Learning tackles. It might be possible to express the problem as a supervised or semi-supervised learning problem, but it would be less natural.
        7. If you don't know how to define the groups, then you can use a clustering algorithm (unsupervised learning) to segment your customers into clusters of similar customers. However, if you know what groups you would like to have, then you can feed many examples of each group to a classification algorithm (supervised learning), and it will classify all your customers into these groups.
        8. Spam detection is a typical supervised learning problem: the algorithm is fed many emails along with their labels (spam or not spam).
        9. An online learning system can learn incrementally, as opposed to a batch learning system. This makes it capable of adapting rapidly to both changing data and autonomous systems, and of training on very large quantities of data.
        10. Out-of-core algorithms can handle vast quantities of data that cannot fit in a computer's main memory. An out-of-core learning algorithm chops the data into mini-batches and uses online learning techniques to learn from these mini-batches.
        11. An instance-based learning system learns the training data by heart; then, when given a new instance, it uses a similarity measure to find the most similar learned instances and uses them to make predictions.
        12. A model has one or more model parameters that determine what it will predict given a new instance (e.g., the slope of a linear model). A learning algorithm tries to find optimal values for these parameters such that the model generalizes well to new instances. A hyperparameter is a parameter of the learning algorithm itself, not of the model (e.g., the amount of regularization to apply).
        13. Model-based learning algorithms search for an optimal value for the model parameters such that the model will generalize well to new instances. We usually train such systems by minimizing a cost function that measures how bad the system is at making predictions on the training data, plus a penalty for model complexity if the model is regularized. To make predictions, we feed the new instance's features into the model's prediction function, using the parameter values found by the learning algorithm.
        14. Some of the main challenges in Machine Learning are the lack of data, poor data quality, nonrepresentative data, uninformative features, excessively simple models that underfit the training data, and excessively complex models that overfit the data.
        15. If a model performs great on the training data but generalizes poorly to new instances, the model is likely overfitting the training data (or we got extremely lucky on the training data). Possible solutions to overfitting are getting more data, simplifying the model (selecting a simpler algorithm, reducing the number of parameters or features used, or regularizing the model), or reducing the noise in the training data.
        16. A test set is used to estimate the generalization error that a model will make on new instances, before the model is launched in production.
        17. A validation set is used to compare models. It makes it possible to select the best model and tune the hyperparameters.
        18. The train-dev set is used when there is a risk of mismatch between the training data and the data used in the validation and test datasets (which should always be as close as possible to the data used once the model is in production). The train-dev set is a part of the training set that's held out (the model is not trained on it). The model is trained on the rest of the training set, and evaluated on both the train-dev set and the validation set. If the model performs well on the training set but not on the train-dev set, then the model is likely overfitting the training set. If it performs well on both the training set and the train-dev set, but not on the validation set, then there is probably a significant data mismatch between the training data and the validation + test data, and you should try to improve the training data to make it look more like the validation + test data.
        19. If you tune hyperparameters using the test set, you risk overfitting the test set, and the generalization error you measure will be optimistic (you may launch a model that performs worse than you expect).
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
