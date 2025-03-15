import marimo

__generated_with = "0.11.20"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Chapter 2 – End-to-end Machine Learning project**
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        *This notebook contains all the sample code and solutions to the exercises in chapter 2.*
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        <table align="left">
          <td>
            <a href="https://colab.research.google.com/github/ageron/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
          </td>
          <td>
            <a target="_blank" href="https://kaggle.com/kernels/welcome?src=https://github.com/ageron/handson-ml3/blob/main/02_end_to_end_machine_learning_project.ipynb"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" /></a>
          </td>
        </table>
        """
    )
    return


@app.cell
def _():
    import duckdb
    import polars as pl
    import altair as alt
    import numpy as np
    from sklearn import set_config

    set_config(transform_output='polars')
    _ = alt.data_transformers.enable("vegafusion")
    _ = alt.renderers.enable("jupyter")
    return alt, duckdb, np, pl, set_config


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Get the Data
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        *Welcome to Machine Learning Housing Corp.! Your task is to predict median house values in Californian districts, given a number of features from these districts.*
        """
    )
    return


@app.cell
def _(duckdb):
    housing = duckdb.read_csv("~/work/handson-ml3/datasets/housing.csv")
    housing.pl()
    return (housing,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Take a Quick Look at the Data Structure
        """
    )
    return


@app.cell
def _(housing):
    housing.describe().pl()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        `ocean_proximity` is a categorical varialble.
        """
    )
    return


@app.cell
def _(housing):
    housing.value_counts("ocean_proximity")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Distribution plots of the numeric variables.
        """
    )
    return


@app.cell
def _(alt, housing):
    housing.pl().plot.bar(
        alt.X(alt.repeat()).bin(maxbins=50),
        alt.Y('count()').title(None)
    ).properties(
        width=200, 
        height=120
    ).repeat(
        repeat=housing.select_types(['double']).columns, 
        columns=3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Create a Test Set
        """
    )
    return


@app.cell
def _(housing):
    from sklearn.model_selection import train_test_split

    train_set, test_set = train_test_split(housing.pl(), test_size=0.2, random_state=42)

    test_set['total_bedrooms'].null_count()
    return test_set, train_set, train_test_split


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To find the probability that a random sample of 1,000 people contains less than 48.5% female or more than 53.5% female when the population's female ratio is 51.1%, we use the [binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution). The `cdf()` method of the binomial distribution gives us the probability that the number of females will be equal or less than the given value.
        """
    )
    return


@app.cell
def _():
    from scipy.stats import binom
    dist = binom(1000, 0.511)
    _prob = dist.cdf(485 - 1) + dist.sf(535)
    print(_prob)
    return binom, dist


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If you prefer simulations over maths, here's how you could get roughly the same result:
        """
    )
    return


@app.cell
def _(dist):
    samples = dist.rvs(size=100000)
    _prob = ((samples < 485) | (samples > 535)).mean()
    print(_prob)
    return (samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We use different bucket boundaries from the book, just for the heck of it.
        """
    )
    return


@app.cell
def _(alt, housing):
    housing_cat = housing.project("*, case "
                    "when median_income < 2.0 then 1"
                    "when median_income < 4.0 then 2"
                    "when median_income < 6.0 then 3"
                    "when median_income < 8.0 then 4"
                    "else 5 end as income_cat"
                )

    housing_cat.aggregate(
        "income_cat, count(1) as districts"
    ).pl().plot.bar(
        alt.X('income_cat:N').axis(labelAngle=0),
        alt.Y('districts')
    ).properties(
        width=300, height=200
    )
    return (housing_cat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To get a single stratified split:
        """
    )
    return


@app.cell
def _(housing_cat, train_test_split):
    strat_train_set, strat_test_set = train_test_split(
        housing_cat.pl(), test_size=0.2, stratify=housing_cat["income_cat"].pl(), random_state=42)
    return strat_test_set, strat_train_set


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Stratified sampling is roughly an order of magnitude more accurate than simple random sampling.
        """
    )
    return


@app.cell
def _(duckdb, housing_cat, strat_test_set):
    duckdb.sql("""
        with overall as (
            select income_cat, count(1) as cnt_overall
            from housing_cat
            group by income_cat),
        rand as (
            select income_cat, count(1) as cnt_rand
            group by income_cat
            using sample 20% (bernoulli, 42)),
        stratified as (
            select income_cat, count(1) as cnt_stratified
            from strat_test_set
            group by income_cat)
        select income_cat, 
            cnt_overall / sum(cnt_overall) over () as "overall %",
            cnt_rand / sum(cnt_rand) over () as "random %",
            cnt_stratified / sum(cnt_stratified) over () as "stratified %",
            "random %" / "overall %" - 1 as "random % error",
            "stratified %" / "overall %" - 1 as "stratified % error"
        from overall inner join rand using(income_cat)
            inner join stratified using(income_cat)
    """).project(
        "income_cat, round(100 * columns(* exclude income_cat), 2)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        That said, the effectiveness of stratified sampling remains to be seen later on.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We won’t use the `income_cat` column again, so might as well drop it.
        """
    )
    return


@app.cell
def _(strat_test_set, strat_train_set):
    for _ in [strat_test_set, strat_train_set]:
        _.drop_in_place('income_cat')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Discover and Visualize the Data to Gain Insights
        """
    )
    return


@app.cell
def _(duckdb, strat_train_set):
    housing_1 = duckdb.sql('from strat_train_set')
    return (housing_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Visualizing Geographical Data
        """
    )
    return


@app.cell
def _(alt, housing_1):
    from vega_datasets import data
    states = alt.topo_feature(data.us_10m.url, feature='states')
    xmin, xmax, ymin, ymax = (-124, -114, 31.5, 42.5)
    alt.layer(alt.Chart(alt.sphere()).mark_geoshape(fill='lightblue', clip=True), alt.Chart(alt.graticule(step=[2, 2])).mark_geoshape(stroke='antiquewhite', strokeWidth=0.5, clip=True), alt.Chart(states).mark_geoshape(fill='beige', stroke='black', clip=True), alt.Chart(housing_1.pl()).mark_circle(opacity=0.7).encode(alt.Color('median_house_value').scale(scheme='turbo'), longitude='longitude', latitude='latitude', size='population')).project(fit={'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [[[xmax, ymax], [xmax, ymin], [xmin, ymin], [xmin, ymax], [xmax, ymax]]]}, 'properties': {}}).properties(width=400, height=400)
    return data, states, xmax, xmin, ymax, ymin


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Looking for Correlations
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's implement `corr_column()` for duckdb tables to calculate the corralations against a given column, in this case `median_house_value`.
        """
    )
    return


@app.cell
def _(duckdb, numeric_tbl):
    def corr_column(tbl: duckdb.DuckDBPyRelation, col: str='median_house_value') -> duckdb.DuckDBPyRelation:
        numeric_tbl = tbl.select_types(['double'])
        return duckdb.sql(f"""
            with long as (
                pivot_longer numeric_tbl on *),
            features as (
                select name as feature, list(value) as val
                from long
                group by all),
            t as (
                select f2.feature, unnest(f1.val) as v1, unnest(f2.val) as v2
                from features as f1 cross join features as f2
                where f1.feature = '{col}')
            select feature, corr(v1, v2)
            from t
            group by all
            order by 2 desc
        """)
    return (corr_column,)


@app.cell
def _(corr_column, housing_1):
    corr_column(housing_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        These numbers roughly agree with the book but not exactly because our training set is randomly chosen.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Scatter matrix plot:
        """
    )
    return


@app.cell
def _(alt, housing_1):
    attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
    alt.Chart(housing_1.pl()).mark_circle(size=1).encode(alt.X(alt.repeat('column'), type='quantitative').axis(labels=False, ticks=False, grid=False), alt.Y(alt.repeat('row'), type='quantitative').axis(labels=False, ticks=False, grid=False)).properties(width=100, height=100).repeat(column=attributes, row=attributes)
    return (attributes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And zoom on median_income, the one with the most conspicuous correlation.
        """
    )
    return


@app.cell
def _(alt, housing_1):
    alt.Chart(housing_1.pl()).mark_circle(opacity=0.1).encode(x='median_income', y='median_house_value').properties(width=400, height=300)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Experimenting with Attribute Combinations
        """
    )
    return


@app.cell
def _(corr_column, housing_1):
    housing_2 = housing_1.project(' *,\n    total_rooms / households as rooms_per_house,\n    total_bedrooms / total_rooms as bedrooms_ratio,\n    population / households as people_per_house\n')
    corr_column(housing_2)
    return (housing_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        - TODO: the correlation with `bedrooms_ratio` (-0.014) is not as big in the book (-0.25).
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Prepare the Data for Machine Learning Algorithms
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's revert to the original training set and separate the target.
        """
    )
    return


@app.cell
def _(duckdb, strat_train_set):
    housing_3 = duckdb.sql('select * exclude median_house_value from strat_train_set')
    housing_labels = duckdb.sql('select median_house_value from strat_train_set')
    return housing_3, housing_labels


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Data Cleaning
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To impute the missing values of `total_bedrooms`.
        """
    )
    return


@app.cell
def _():
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median")
    return SimpleImputer, imputer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Separating out the numerical attributes to use the `"median"` strategy (as it cannot be calculated on text attributes like `ocean_proximity`), then transform the training set.
        """
    )
    return


@app.cell
def _(housing_3, imputer):
    X = imputer.fit_transform(housing_3.select_types(['double']).pl())
    X['total_bedrooms'].null_count()
    return (X,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Check that this is the same as manually computing the median of each attribute:
        """
    )
    return


@app.cell
def _(housing_3, imputer):
    print(imputer.statistics_)
    housing_3.select_types(['double']).aggregate('median(columns(*))').show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's drop some outliers:
        """
    )
    return


@app.cell
def _(X):
    from sklearn.ensemble import IsolationForest

    isolation_forest = IsolationForest(random_state=42)
    outlier_pred = isolation_forest.fit_predict(X)
    outlier_pred
    return IsolationForest, isolation_forest, outlier_pred


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To drop outliers:
        """
    )
    return


@app.cell
def _(housing_3, outlier_pred, pl):
    housing_3.pl().hstack([pl.Series('outlier', outlier_pred)]).filter(pl.col('outlier') == 1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Handling Text and Categorical Attributes
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's preprocess the categorical input feature, `ocean_proximity`:
        """
    )
    return


@app.cell
def _(housing_3):
    housing_3['ocean_proximity']
    return


@app.cell
def _(housing_3):
    from sklearn.preprocessing import OrdinalEncoder
    ordinal_encoder = OrdinalEncoder()
    ordinal_encoder.fit_transform(housing_3['ocean_proximity'].pl())
    return OrdinalEncoder, ordinal_encoder


@app.cell
def _(ordinal_encoder):
    ordinal_encoder.categories_
    return


@app.cell
def _(housing_3):
    from sklearn.preprocessing import OneHotEncoder
    cat_encoder = OneHotEncoder(sparse_output=False)
    housing_cat_1hot = cat_encoder.fit_transform(housing_3['ocean_proximity'].pl())
    housing_cat_1hot
    return OneHotEncoder, cat_encoder, housing_cat_1hot


@app.cell
def _(cat_encoder):
    cat_encoder.categories_
    return


@app.cell
def _(cat_encoder):
    cat_encoder.feature_names_in_
    return


@app.cell
def _(cat_encoder):
    cat_encoder.get_feature_names_out()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Prefer `OneHotEncoder` to pandas' `get_dummies()`.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Feature Scaling
        """
    )
    return


@app.cell
def _(housing_num):
    from sklearn.preprocessing import MinMaxScaler

    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)
    return MinMaxScaler, housing_num_min_max_scaled, min_max_scaler


@app.cell
def _(housing_num):
    from sklearn.preprocessing import StandardScaler

    std_scaler = StandardScaler()
    housing_num_std_scaled = std_scaler.fit_transform(housing_num)
    return StandardScaler, housing_num_std_scaled, std_scaler


@app.cell
def _(housing_3, np, plt, save_fig):
    _fig, axs = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    housing_3['population'].hist(ax=axs[0], bins=50)
    housing_3['population'].apply(np.log).hist(ax=axs[1], bins=50)
    axs[0].set_xlabel('Population')
    axs[1].set_xlabel('Log of population')
    axs[0].set_ylabel('Number of districts')
    save_fig('long_tail_plot')
    plt.show()
    return (axs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        What if we replace each value with its percentile?
        """
    )
    return


@app.cell
def _(housing_3, np, pd, plt):
    percentiles = [np.percentile(housing_3['median_income'], p) for p in range(1, 100)]
    flattened_median_income = pd.cut(housing_3['median_income'], bins=[-np.inf] + percentiles + [np.inf], labels=range(1, 100 + 1))
    flattened_median_income.hist(bins=50)
    plt.xlabel('Median income percentile')
    plt.ylabel('Number of districts')
    plt.show()
    return flattened_median_income, percentiles


@app.cell
def _(housing_3):
    from sklearn.metrics.pairwise import rbf_kernel
    age_simil_35 = rbf_kernel(housing_3[['housing_median_age']], [[35]], gamma=0.1)
    return age_simil_35, rbf_kernel


@app.cell
def _(housing_3, np, plt, rbf_kernel, save_fig):
    ages = np.linspace(housing_3['housing_median_age'].min(), housing_3['housing_median_age'].max(), 500).reshape(-1, 1)
    gamma1 = 0.1
    gamma2 = 0.03
    rbf1 = rbf_kernel(ages, [[35]], gamma=gamma1)
    rbf2 = rbf_kernel(ages, [[35]], gamma=gamma2)
    _fig, ax1 = plt.subplots()
    ax1.set_xlabel('Housing median age')
    ax1.set_ylabel('Number of districts')
    ax1.hist(housing_3['housing_median_age'], bins=50)
    ax2 = ax1.twinx()
    color = 'blue'
    ax2.plot(ages, rbf1, color=color, label='gamma = 0.10')
    ax2.plot(ages, rbf2, color=color, label='gamma = 0.03', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel('Age similarity', color=color)
    plt.legend(loc='upper left')
    save_fig('age_similarity_plot')
    plt.show()
    return ages, ax1, ax2, color, gamma1, gamma2, rbf1, rbf2


@app.cell
def _(StandardScaler, housing_3, housing_labels):
    from sklearn.linear_model import LinearRegression
    target_scaler = StandardScaler()
    scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())
    _model = LinearRegression()
    _model.fit(housing_3[['median_income']], scaled_labels)
    some_new_data = housing_3[['median_income']].iloc[:5]
    scaled_predictions = _model.predict(some_new_data)
    predictions = target_scaler.inverse_transform(scaled_predictions)
    return (
        LinearRegression,
        predictions,
        scaled_labels,
        scaled_predictions,
        some_new_data,
        target_scaler,
    )


@app.cell
def _(predictions):
    predictions
    return


@app.cell
def _(
    LinearRegression,
    StandardScaler,
    housing_3,
    housing_labels,
    some_new_data,
):
    from sklearn.compose import TransformedTargetRegressor
    _model = TransformedTargetRegressor(LinearRegression(), transformer=StandardScaler())
    _model.fit(housing_3[['median_income']], housing_labels)
    predictions_1 = _model.predict(some_new_data)
    return TransformedTargetRegressor, predictions_1


@app.cell
def _(predictions_1):
    predictions_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Custom Transformers
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To create simple transformers:
        """
    )
    return


@app.cell
def _(housing_3, np):
    from sklearn.preprocessing import FunctionTransformer
    log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
    log_pop = log_transformer.transform(housing_3[['population']])
    return FunctionTransformer, log_pop, log_transformer


@app.cell
def _(FunctionTransformer, housing_3, rbf_kernel):
    rbf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[[35.0]], gamma=0.1))
    age_simil_35_1 = rbf_transformer.transform(housing_3[['housing_median_age']])
    return age_simil_35_1, rbf_transformer


@app.cell
def _(age_simil_35_1):
    age_simil_35_1
    return


@app.cell
def _(FunctionTransformer, housing_3, rbf_kernel):
    sf_coords = (37.7749, -122.41)
    sf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[sf_coords], gamma=0.1))
    sf_simil = sf_transformer.transform(housing_3[['latitude', 'longitude']])
    return sf_coords, sf_simil, sf_transformer


@app.cell
def _(sf_simil):
    sf_simil
    return


@app.cell
def _(FunctionTransformer, np):
    ratio_transformer = FunctionTransformer(lambda X: X[:, [0]] / X[:, [1]])
    ratio_transformer.transform(np.array([[1., 2.], [3., 4.]]))
    return (ratio_transformer,)


@app.cell
def _():
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.utils.validation import check_array, check_is_fitted

    class StandardScalerClone(BaseEstimator, TransformerMixin):
        def __init__(self, with_mean=True):  # no *args or **kwargs!
            self.with_mean = with_mean

        def fit(self, X, y=None):  # y is required even though we don't use it
            X = check_array(X)  # checks that X is an array with finite float values
            self.mean_ = X.mean(axis=0)
            self.scale_ = X.std(axis=0)
            self.n_features_in_ = X.shape[1]  # every estimator stores this in fit()
            return self  # always return self!

        def transform(self, X):
            check_is_fitted(self)  # looks for learned attributes (with trailing _)
            X = check_array(X)
            assert self.n_features_in_ == X.shape[1]
            if self.with_mean:
                X = X - self.mean_
            return X / self.scale_
    return (
        BaseEstimator,
        StandardScalerClone,
        TransformerMixin,
        check_array,
        check_is_fitted,
    )


@app.cell
def _(BaseEstimator, TransformerMixin, rbf_kernel):
    from sklearn.cluster import KMeans

    class ClusterSimilarity(BaseEstimator, TransformerMixin):
        def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
            self.n_clusters = n_clusters
            self.gamma = gamma
            self.random_state = random_state

        def fit(self, X, y=None, sample_weight=None):
            self.kmeans_ = KMeans(self.n_clusters, n_init=10,
                                  random_state=self.random_state)
            self.kmeans_.fit(X, sample_weight=sample_weight)
            return self  # always return self!

        def transform(self, X):
            return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
        def get_feature_names_out(self, names=None):
            return [f"Cluster {i} similarity" for i in range(self.n_clusters)]
    return ClusterSimilarity, KMeans


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Warning**:
        * There was a change in Scikit-Learn 1.3.0 which affected the random number generator for `KMeans` initialization. Therefore the results will be different than in the book if you use Scikit-Learn ≥ 1.3. That's not a problem as long as you don't expect the outputs to be perfectly identical.
        * Throughout this notebook, when `n_init` was not set when creating a `KMeans` estimator, I explicitly set it to `n_init=10` to avoid a warning about the fact that the default value for this hyperparameter will change from 10 to `"auto"` in Scikit-Learn 1.4.
        """
    )
    return


@app.cell
def _(ClusterSimilarity, housing_3, housing_labels):
    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)
    similarities = cluster_simil.fit_transform(housing_3[['latitude', 'longitude']], sample_weight=housing_labels)
    return cluster_simil, similarities


@app.cell
def _(similarities):
    similarities[:3].round(2)
    return


@app.cell
def _(cluster_simil, housing_3, plt, save_fig, similarities):
    housing_renamed = housing_3.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude', 'population': 'Population', 'median_house_value': 'Median house value (ᴜsᴅ)'})
    housing_renamed['Max cluster similarity'] = similarities.max(axis=1)
    housing_renamed.plot(kind='scatter', x='Longitude', y='Latitude', grid=True, s=housing_renamed['Population'] / 100, label='Population', c='Max cluster similarity', cmap='jet', colorbar=True, legend=True, sharex=False, figsize=(10, 7))
    plt.plot(cluster_simil.kmeans_.cluster_centers_[:, 1], cluster_simil.kmeans_.cluster_centers_[:, 0], linestyle='', color='black', marker='X', markersize=20, label='Cluster centers')
    plt.legend(loc='upper right')
    save_fig('district_cluster_plot')
    plt.show()
    return (housing_renamed,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Transformation Pipelines
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's build a pipeline to preprocess the numerical attributes:
        """
    )
    return


@app.cell
def _(SimpleImputer, StandardScaler):
    from sklearn.pipeline import Pipeline

    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("standardize", StandardScaler()),
    ])
    return Pipeline, num_pipeline


@app.cell
def _(SimpleImputer, StandardScaler):
    from sklearn.pipeline import make_pipeline
    num_pipeline_1 = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
    return make_pipeline, num_pipeline_1


@app.cell
def _(num_pipeline_1, set_config):
    set_config(display='diagram')
    num_pipeline_1
    return


@app.cell
def _(housing_num, num_pipeline_1):
    housing_num_prepared = num_pipeline_1.fit_transform(housing_num)
    housing_num_prepared[:2].round(2)
    return (housing_num_prepared,)


@app.cell
def _(SimpleImputer):
    def monkey_patch_get_signature_names_out():
        """Monkey patch some classes which did not handle get_feature_names_out()
           correctly in Scikit-Learn 1.0.*."""
        from inspect import Signature, signature, Parameter
        import pandas as pd
        from sklearn.pipeline import make_pipeline, Pipeline
        from sklearn.preprocessing import FunctionTransformer, StandardScaler

        default_get_feature_names_out = StandardScaler.get_feature_names_out

        if not hasattr(SimpleImputer, "get_feature_names_out"):
          print("Monkey-patching SimpleImputer.get_feature_names_out()")
          SimpleImputer.get_feature_names_out = default_get_feature_names_out

        if not hasattr(FunctionTransformer, "get_feature_names_out"):
            print("Monkey-patching FunctionTransformer.get_feature_names_out()")
            orig_init = FunctionTransformer.__init__
            orig_sig = signature(orig_init)

            def __init__(*args, feature_names_out=None, **kwargs):
                orig_sig.bind(*args, **kwargs)
                orig_init(*args, **kwargs)
                args[0].feature_names_out = feature_names_out

            __init__.__signature__ = Signature(
                list(signature(orig_init).parameters.values()) + [
                    Parameter("feature_names_out", Parameter.KEYWORD_ONLY)])

            def get_feature_names_out(self, names=None):
                if callable(self.feature_names_out):
                    return self.feature_names_out(self, names)
                assert self.feature_names_out == "one-to-one"
                return default_get_feature_names_out(self, names)

            FunctionTransformer.__init__ = __init__
            FunctionTransformer.get_feature_names_out = get_feature_names_out

    monkey_patch_get_signature_names_out()
    return (monkey_patch_get_signature_names_out,)


@app.cell
def _(housing_num, housing_num_prepared, num_pipeline_1, pd):
    df_housing_num_prepared = pd.DataFrame(housing_num_prepared, columns=num_pipeline_1.get_feature_names_out(), index=housing_num.index)
    return (df_housing_num_prepared,)


@app.cell
def _(df_housing_num_prepared):
    df_housing_num_prepared.head(2)  # extra code
    return


@app.cell
def _(num_pipeline_1):
    num_pipeline_1.steps
    return


@app.cell
def _(num_pipeline_1):
    num_pipeline_1[1]
    return


@app.cell
def _(num_pipeline_1):
    num_pipeline_1[:-1]
    return


@app.cell
def _(num_pipeline_1):
    num_pipeline_1.named_steps['simpleimputer']
    return


@app.cell
def _(num_pipeline_1):
    num_pipeline_1.set_params(simpleimputer__strategy='median')
    return


@app.cell
def _(OneHotEncoder, SimpleImputer, make_pipeline, num_pipeline_1):
    from sklearn.compose import ColumnTransformer
    num_attribs = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']
    cat_attribs = ['ocean_proximity']
    cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
    preprocessing = ColumnTransformer([('num', num_pipeline_1, num_attribs), ('cat', cat_pipeline, cat_attribs)])
    return (
        ColumnTransformer,
        cat_attribs,
        cat_pipeline,
        num_attribs,
        preprocessing,
    )


@app.cell
def _(cat_pipeline, np, num_pipeline_1):
    from sklearn.compose import make_column_selector, make_column_transformer
    preprocessing_1 = make_column_transformer((num_pipeline_1, make_column_selector(dtype_include=np.number)), (cat_pipeline, make_column_selector(dtype_include=object)))
    return make_column_selector, make_column_transformer, preprocessing_1


@app.cell
def _(housing_3, preprocessing_1):
    housing_prepared = preprocessing_1.fit_transform(housing_3)
    return (housing_prepared,)


@app.cell
def _(housing_3, housing_prepared, pd, preprocessing_1):
    housing_prepared_fr = pd.DataFrame(housing_prepared, columns=preprocessing_1.get_feature_names_out(), index=housing_3.index)
    housing_prepared_fr.head(2)
    return (housing_prepared_fr,)


@app.cell
def _(
    ClusterSimilarity,
    ColumnTransformer,
    FunctionTransformer,
    SimpleImputer,
    StandardScaler,
    cat_pipeline,
    make_column_selector,
    make_pipeline,
    np,
):
    def _column_ratio(X):
        return X[:, [0]] / X[:, [1]]

    def ratio_name(function_transformer, feature_names_in):
        return ['ratio']

    def ratio_pipeline():
        return make_pipeline(SimpleImputer(strategy='median'), FunctionTransformer(_column_ratio, feature_names_out=ratio_name), StandardScaler())
    log_pipeline = make_pipeline(SimpleImputer(strategy='median'), FunctionTransformer(np.log, feature_names_out='one-to-one'), StandardScaler())
    cluster_simil_1 = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)
    default_num_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
    preprocessing_2 = ColumnTransformer([('bedrooms', ratio_pipeline(), ['total_bedrooms', 'total_rooms']), ('rooms_per_house', ratio_pipeline(), ['total_rooms', 'households']), ('people_per_house', ratio_pipeline(), ['population', 'households']), ('log', log_pipeline, ['total_bedrooms', 'total_rooms', 'population', 'households', 'median_income']), ('geo', cluster_simil_1, ['latitude', 'longitude']), ('cat', cat_pipeline, make_column_selector(dtype_include=object))], remainder=default_num_pipeline)
    return (
        cluster_simil_1,
        default_num_pipeline,
        log_pipeline,
        preprocessing_2,
        ratio_name,
        ratio_pipeline,
    )


@app.cell
def _(housing_3, preprocessing_2):
    housing_prepared_1 = preprocessing_2.fit_transform(housing_3)
    housing_prepared_1.shape
    return (housing_prepared_1,)


@app.cell
def _(preprocessing_2):
    preprocessing_2.get_feature_names_out()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Select and Train a Model
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Training and Evaluating on the Training Set
        """
    )
    return


@app.cell
def _(
    LinearRegression,
    housing_3,
    housing_labels,
    make_pipeline,
    preprocessing_2,
):
    lin_reg = make_pipeline(preprocessing_2, LinearRegression())
    lin_reg.fit(housing_3, housing_labels)
    return (lin_reg,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's try the full preprocessing pipeline on a few training instances:
        """
    )
    return


@app.cell
def _(housing_3, lin_reg):
    housing_predictions = lin_reg.predict(housing_3)
    housing_predictions[:5].round(-2)
    return (housing_predictions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Compare against the actual values:
        """
    )
    return


@app.cell
def _(housing_labels):
    housing_labels.iloc[:5].values
    return


@app.cell
def _(housing_labels, housing_predictions):
    # extra code – computes the error ratios discussed in the book
    error_ratios = housing_predictions[:5].round(-2) / housing_labels.iloc[:5].values - 1
    print(", ".join([f"{100 * ratio:.1f}%" for ratio in error_ratios]))
    return (error_ratios,)


@app.cell
def _(housing_labels, housing_predictions):
    from sklearn.metrics import mean_squared_error

    lin_rmse = mean_squared_error(housing_labels, housing_predictions,
                                  squared=False)
    lin_rmse
    return lin_rmse, mean_squared_error


@app.cell
def _(housing_3, housing_labels, make_pipeline, preprocessing_2):
    from sklearn.tree import DecisionTreeRegressor
    tree_reg = make_pipeline(preprocessing_2, DecisionTreeRegressor(random_state=42))
    tree_reg.fit(housing_3, housing_labels)
    return DecisionTreeRegressor, tree_reg


@app.cell
def _(housing_3, housing_labels, mean_squared_error, tree_reg):
    housing_predictions_1 = tree_reg.predict(housing_3)
    tree_rmse = mean_squared_error(housing_labels, housing_predictions_1, squared=False)
    tree_rmse
    return housing_predictions_1, tree_rmse


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Better Evaluation Using Cross-Validation
        """
    )
    return


@app.cell
def _(housing_3, housing_labels, tree_reg):
    from sklearn.model_selection import cross_val_score
    tree_rmses = -cross_val_score(tree_reg, housing_3, housing_labels, scoring='neg_root_mean_squared_error', cv=10)
    return cross_val_score, tree_rmses


@app.cell
def _(pd, tree_rmses):
    pd.Series(tree_rmses).describe()
    return


@app.cell
def _(cross_val_score, housing_3, housing_labels, lin_reg, pd):
    lin_rmses = -cross_val_score(lin_reg, housing_3, housing_labels, scoring='neg_root_mean_squared_error', cv=10)
    pd.Series(lin_rmses).describe()
    return (lin_rmses,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Warning:** the following cell may take a few minutes to run:
        """
    )
    return


@app.cell
def _(
    cross_val_score,
    housing_3,
    housing_labels,
    make_pipeline,
    preprocessing_2,
):
    from sklearn.ensemble import RandomForestRegressor
    forest_reg = make_pipeline(preprocessing_2, RandomForestRegressor(random_state=42))
    forest_rmses = -cross_val_score(forest_reg, housing_3, housing_labels, scoring='neg_root_mean_squared_error', cv=10)
    return RandomForestRegressor, forest_reg, forest_rmses


@app.cell
def _(forest_rmses, pd):
    pd.Series(forest_rmses).describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's compare this RMSE measured using cross-validation (the "validation error") with the RMSE measured on the training set (the "training error"):
        """
    )
    return


@app.cell
def _(forest_reg, housing_3, housing_labels, mean_squared_error):
    forest_reg.fit(housing_3, housing_labels)
    housing_predictions_2 = forest_reg.predict(housing_3)
    forest_rmse = mean_squared_error(housing_labels, housing_predictions_2, squared=False)
    forest_rmse
    return forest_rmse, housing_predictions_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The training error is much lower than the validation error, which usually means that the model has overfit the training set. Another possible explanation may be that there's a mismatch between the training data and the validation data, but it's not the case here, since both came from the same dataset that we shuffled and split in two parts.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Fine-Tune Your Model
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Grid Search
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Warning:** the following cell may take a few minutes to run:
        """
    )
    return


@app.cell
def _(
    Pipeline,
    RandomForestRegressor,
    housing_3,
    housing_labels,
    preprocessing_2,
):
    from sklearn.model_selection import GridSearchCV
    full_pipeline = Pipeline([('preprocessing', preprocessing_2), ('random_forest', RandomForestRegressor(random_state=42))])
    _param_grid = [{'preprocessing__geo__n_clusters': [5, 8, 10], 'random_forest__max_features': [4, 6, 8]}, {'preprocessing__geo__n_clusters': [10, 15], 'random_forest__max_features': [6, 8, 10]}]
    grid_search = GridSearchCV(full_pipeline, _param_grid, cv=3, scoring='neg_root_mean_squared_error')
    grid_search.fit(housing_3, housing_labels)
    return GridSearchCV, full_pipeline, grid_search


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        You can get the full list of hyperparameters available for tuning by looking at `full_pipeline.get_params().keys()`:
        """
    )
    return


@app.cell
def _(full_pipeline):
    # extra code – shows part of the output of get_params().keys()
    print(str(full_pipeline.get_params().keys())[:1000] + "...")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The best hyperparameter combination found:
        """
    )
    return


@app.cell
def _(grid_search):
    grid_search.best_params_
    return


@app.cell
def _(grid_search):
    grid_search.best_estimator_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's look at the score of each hyperparameter combination tested during the grid search:
        """
    )
    return


@app.cell
def _(grid_search, np, pd):
    _cv_res = pd.DataFrame(grid_search.cv_results_)
    _cv_res.sort_values(by='mean_test_score', ascending=False, inplace=True)
    _cv_res = _cv_res[['param_preprocessing__geo__n_clusters', 'param_random_forest__max_features', 'split0_test_score', 'split1_test_score', 'split2_test_score', 'mean_test_score']]
    score_cols = ['split0', 'split1', 'split2', 'mean_test_rmse']
    _cv_res.columns = ['n_clusters', 'max_features'] + score_cols
    _cv_res[score_cols] = -_cv_res[score_cols].round().astype(np.int64)
    _cv_res.head()
    return (score_cols,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Randomized Search
        """
    )
    return


@app.cell
def _():
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingRandomSearchCV
    return HalvingRandomSearchCV, enable_halving_search_cv


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Try 30 (`n_iter` × `cv`) random combinations of hyperparameters:
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Warning:** the following cell may take a few minutes to run:
        """
    )
    return


@app.cell
def _(full_pipeline, housing_3, housing_labels):
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint
    _param_distribs = {'preprocessing__geo__n_clusters': _randint(low=3, high=50), 'random_forest__max_features': _randint(low=2, high=20)}
    rnd_search = RandomizedSearchCV(full_pipeline, param_distributions=_param_distribs, n_iter=10, cv=3, scoring='neg_root_mean_squared_error', random_state=42)
    rnd_search.fit(housing_3, housing_labels)
    return RandomizedSearchCV, randint, rnd_search


@app.cell
def _(np, pd, rnd_search, score_cols):
    _cv_res = pd.DataFrame(rnd_search.cv_results_)
    _cv_res.sort_values(by='mean_test_score', ascending=False, inplace=True)
    _cv_res = _cv_res[['param_preprocessing__geo__n_clusters', 'param_random_forest__max_features', 'split0_test_score', 'split1_test_score', 'split2_test_score', 'mean_test_score']]
    _cv_res.columns = ['n_clusters', 'max_features'] + score_cols
    _cv_res[score_cols] = -_cv_res[score_cols].round().astype(np.int64)
    _cv_res.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Bonus section: how to choose the sampling distribution for a hyperparameter**

        * `scipy.stats.randint(a, b+1)`: for hyperparameters with _discrete_ values that range from a to b, and all values in that range seem equally likely.
        * `scipy.stats.uniform(a, b)`: this is very similar, but for _continuous_ hyperparameters.
        * `scipy.stats.geom(1 / scale)`: for discrete values, when you want to sample roughly in a given scale. E.g., with scale=1000 most samples will be in this ballpark, but ~10% of all samples will be <100 and ~10% will be >2300.
        * `scipy.stats.expon(scale)`: this is the continuous equivalent of `geom`. Just set `scale` to the most likely value.
        * `scipy.stats.loguniform(a, b)`: when you have almost no idea what the optimal hyperparameter value's scale is. If you set a=0.01 and b=100, then you're just as likely to sample a value between 0.01 and 0.1 as a value between 10 and 100.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here are plots of the probability mass functions (for discrete variables), and probability density functions (for continuous variables) for `randint()`, `uniform()`, `geom()` and `expon()`:
        """
    )
    return


@app.cell
def _(np, plt):
    from scipy.stats import randint, uniform, geom, expon
    _xs1 = np.arange(0, 7 + 1)
    randint_distrib = _randint(0, 7 + 1).pmf(_xs1)
    xs2 = np.linspace(0, 7, 500)
    uniform_distrib = uniform(0, 7).pdf(xs2)
    _xs3 = np.arange(0, 7 + 1)
    geom_distrib = geom(0.5).pmf(_xs3)
    xs4 = np.linspace(0, 7, 500)
    _expon_distrib = expon(scale=1).pdf(xs4)
    plt.figure(figsize=(12, 7))
    plt.subplot(2, 2, 1)
    plt.bar(_xs1, randint_distrib, label='scipy.randint(0, 7 + 1)')
    plt.ylabel('Probability')
    plt.legend()
    plt.axis([-1, 8, 0, 0.2])
    plt.subplot(2, 2, 2)
    plt.fill_between(xs2, uniform_distrib, label='scipy.uniform(0, 7)')
    plt.ylabel('PDF')
    plt.legend()
    plt.axis([-1, 8, 0, 0.2])
    plt.subplot(2, 2, 3)
    plt.bar(_xs3, geom_distrib, label='scipy.geom(0.5)')
    plt.xlabel('Hyperparameter value')
    plt.ylabel('Probability')
    plt.legend()
    plt.axis([0, 7, 0, 1])
    plt.subplot(2, 2, 4)
    plt.fill_between(xs4, _expon_distrib, label='scipy.expon(scale=1)')
    plt.xlabel('Hyperparameter value')
    plt.ylabel('PDF')
    plt.legend()
    plt.axis([0, 7, 0, 1])
    plt.show()
    return (
        expon,
        geom,
        geom_distrib,
        randint,
        randint_distrib,
        uniform,
        uniform_distrib,
        xs2,
        xs4,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here are the PDF for `expon()` and `loguniform()` (left column), as well as the PDF of log(X) (right column). The right column shows the distribution of hyperparameter _scales_. You can see that `expon()` favors hyperparameters with roughly the desired scale, with a longer tail towards the smaller scales. But `loguniform()` does not favor any scale, they are all equally likely:
        """
    )
    return


@app.cell
def _(expon, np, plt, uniform):
    from scipy.stats import loguniform
    _xs1 = np.linspace(0, 7, 500)
    _expon_distrib = expon(scale=1).pdf(_xs1)
    log_xs2 = np.linspace(-5, 3, 500)
    log_expon_distrib = np.exp(log_xs2 - np.exp(log_xs2))
    _xs3 = np.linspace(0.001, 1000, 500)
    loguniform_distrib = loguniform(0.001, 1000).pdf(_xs3)
    log_xs4 = np.linspace(np.log(0.001), np.log(1000), 500)
    log_loguniform_distrib = uniform(np.log(0.001), np.log(1000)).pdf(log_xs4)
    plt.figure(figsize=(12, 7))
    plt.subplot(2, 2, 1)
    plt.fill_between(_xs1, _expon_distrib, label='scipy.expon(scale=1)')
    plt.ylabel('PDF')
    plt.legend()
    plt.axis([0, 7, 0, 1])
    plt.subplot(2, 2, 2)
    plt.fill_between(log_xs2, log_expon_distrib, label='log(X) with X ~ expon')
    plt.legend()
    plt.axis([-5, 3, 0, 1])
    plt.subplot(2, 2, 3)
    plt.fill_between(_xs3, loguniform_distrib, label='scipy.loguniform(0.001, 1000)')
    plt.xlabel('Hyperparameter value')
    plt.ylabel('PDF')
    plt.legend()
    plt.axis([0.001, 1000, 0, 0.005])
    plt.subplot(2, 2, 4)
    plt.fill_between(log_xs4, log_loguniform_distrib, label='log(X) with X ~ loguniform')
    plt.xlabel('Log of hyperparameter value')
    plt.legend()
    plt.axis([-8, 1, 0, 0.2])
    plt.show()
    return (
        log_expon_distrib,
        log_loguniform_distrib,
        log_xs2,
        log_xs4,
        loguniform,
        loguniform_distrib,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Analyze the Best Models and Their Errors
        """
    )
    return


@app.cell
def _(rnd_search):
    final_model = rnd_search.best_estimator_  # includes preprocessing
    feature_importances = final_model["random_forest"].feature_importances_
    feature_importances.round(2)
    return feature_importances, final_model


@app.cell
def _(feature_importances, final_model):
    sorted(zip(feature_importances,
               final_model["preprocessing"].get_feature_names_out()),
               reverse=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Evaluate Your System on the Test Set
        """
    )
    return


@app.cell
def _(final_model, mean_squared_error, strat_test_set):
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    final_predictions = final_model.predict(X_test)

    final_rmse = mean_squared_error(y_test, final_predictions, squared=False)
    print(final_rmse)
    return X_test, final_predictions, final_rmse, y_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can compute a 95% confidence interval for the test RMSE:
        """
    )
    return


@app.cell
def _(final_predictions, np, y_test):
    from scipy import stats

    confidence = 0.95
    squared_errors = (final_predictions - y_test) ** 2
    np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                             loc=squared_errors.mean(),
                             scale=stats.sem(squared_errors)))
    return confidence, squared_errors, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We could compute the interval manually like this:
        """
    )
    return


@app.cell
def _(confidence, np, squared_errors, stats):
    # extra code – shows how to compute a confidence interval for the RMSE
    m = len(squared_errors)
    mean = squared_errors.mean()
    tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
    tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
    np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)
    return m, mean, tmargin, tscore


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Alternatively, we could use a z-score rather than a t-score. Since the test set is not too small, it won't make a big difference:
        """
    )
    return


@app.cell
def _(confidence, m, mean, np, squared_errors, stats):
    # extra code – computes a confidence interval again using a z-score
    zscore = stats.norm.ppf((1 + confidence) / 2)
    zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
    np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)
    return zmargin, zscore


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Model persistence using joblib
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Save the final model:
        """
    )
    return


@app.cell
def _(final_model):
    import joblib

    joblib.dump(final_model, "my_california_housing_model.pkl")
    return (joblib,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now you can deploy this model to production. For example, the following code could be a script that would run in production:
        """
    )
    return


@app.cell
def _(housing_3, joblib):
    def _column_ratio(X):
        return X[:, [0]] / X[:, [1]]
    final_model_reloaded = joblib.load('my_california_housing_model.pkl')
    new_data = housing_3.iloc[:5]
    predictions_2 = final_model_reloaded.predict(new_data)
    return final_model_reloaded, new_data, predictions_2


@app.cell
def _(predictions_2):
    predictions_2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        You could use pickle instead, but joblib is more efficient.
        """
    )
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
        ## 1.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Exercise: _Try a Support Vector Machine regressor (`sklearn.svm.SVR`) with various hyperparameters, such as `kernel="linear"` (with various values for the `C` hyperparameter) or `kernel="rbf"` (with various values for the `C` and `gamma` hyperparameters). Note that SVMs don't scale well to large datasets, so you should probably train your model on just the first 5,000 instances of the training set and use only 3-fold cross-validation, or else it will take hours. Don't worry about what the hyperparameters mean for now (see the SVM notebook if you're interested). How does the best `SVR` predictor perform?_
        """
    )
    return


@app.cell
def _(GridSearchCV, Pipeline, housing_3, housing_labels, preprocessing_2):
    from sklearn.svm import SVR
    _param_grid = [{'svr__kernel': ['linear'], 'svr__C': [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0]}, {'svr__kernel': ['rbf'], 'svr__C': [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0], 'svr__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}]
    svr_pipeline = Pipeline([('preprocessing', preprocessing_2), ('svr', SVR())])
    grid_search_1 = GridSearchCV(svr_pipeline, _param_grid, cv=3, scoring='neg_root_mean_squared_error')
    grid_search_1.fit(housing_3.iloc[:5000], housing_labels.iloc[:5000])
    return SVR, grid_search_1, svr_pipeline


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The best model achieves the following score (evaluated using 3-fold cross validation):
        """
    )
    return


@app.cell
def _(grid_search_1):
    svr_grid_search_rmse = -grid_search_1.best_score_
    svr_grid_search_rmse
    return (svr_grid_search_rmse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        That's much worse than the `RandomForestRegressor` (but to be fair, we trained the model on much less data). Let's check the best hyperparameters found:
        """
    )
    return


@app.cell
def _(grid_search_1):
    grid_search_1.best_params_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The linear kernel seems better than the RBF kernel. Notice that the value of `C` is the maximum tested value. When this happens you definitely want to launch the grid search again with higher values for `C` (removing the smallest values), because it is likely that higher values of `C` will be better.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Exercise: _Try replacing the `GridSearchCV` with a `RandomizedSearchCV`._
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Warning:** the following cell will take several minutes to run. You can specify `verbose=2` when creating the `RandomizedSearchCV` if you want to see the training details.
        """
    )
    return


@app.cell
def _(
    RandomizedSearchCV,
    expon_1,
    housing_3,
    housing_labels,
    loguniform_1,
    svr_pipeline,
):
    from scipy.stats import expon, loguniform
    _param_distribs = {'svr__kernel': ['linear', 'rbf'], 'svr__C': loguniform_1(20, 200000), 'svr__gamma': expon_1(scale=1.0)}
    rnd_search_1 = RandomizedSearchCV(svr_pipeline, param_distributions=_param_distribs, n_iter=50, cv=3, scoring='neg_root_mean_squared_error', random_state=42)
    rnd_search_1.fit(housing_3.iloc[:5000], housing_labels.iloc[:5000])
    return expon, loguniform, rnd_search_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The best model achieves the following score (evaluated using 3-fold cross validation):
        """
    )
    return


@app.cell
def _(rnd_search_1):
    svr_rnd_search_rmse = -rnd_search_1.best_score_
    svr_rnd_search_rmse
    return (svr_rnd_search_rmse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now that's really much better, but still far from the `RandomForestRegressor`'s performance. Let's check the best hyperparameters found:
        """
    )
    return


@app.cell
def _(rnd_search_1):
    rnd_search_1.best_params_
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This time the search found a good set of hyperparameters for the RBF kernel. Randomized search tends to find better hyperparameters than grid search in the same amount of time.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note that we used the `expon()` distribution for `gamma`, with a scale of 1, so `RandomSearch` mostly searched for values roughly of that scale: about 80% of the samples were between 0.1 and 2.3 (roughly 10% were smaller and 10% were larger):
        """
    )
    return


@app.cell
def _(expon_1, np):
    np.random.seed(42)
    s = expon_1(scale=1).rvs(100000)
    ((s > 0.105) & (s < 2.29)).sum() / 100000
    return (s,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We used the `loguniform()` distribution for `C`, meaning we did not have a clue what the optimal scale of `C` was before running the random search. It explored the range from 20 to 200 just as much as the range from 2,000 to 20,000 or from 20,000 to 200,000.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 3.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Exercise: _Try adding a `SelectFromModel` transformer in the preparation pipeline to select only the most important attributes._
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's create a new pipeline that runs the previously defined preparation pipeline, and adds a `SelectFromModel` transformer based on a `RandomForestRegressor` before the final regressor:
        """
    )
    return


@app.cell
def _(Pipeline, RandomForestRegressor, SVR, preprocessing_2, rnd_search_1):
    from sklearn.feature_selection import SelectFromModel
    selector_pipeline = Pipeline([('preprocessing', preprocessing_2), ('selector', SelectFromModel(RandomForestRegressor(random_state=42), threshold=0.005)), ('svr', SVR(C=rnd_search_1.best_params_['svr__C'], gamma=rnd_search_1.best_params_['svr__gamma'], kernel=rnd_search_1.best_params_['svr__kernel']))])
    return SelectFromModel, selector_pipeline


@app.cell
def _(cross_val_score, housing_3, housing_labels, pd, selector_pipeline):
    selector_rmses = -cross_val_score(selector_pipeline, housing_3.iloc[:5000], housing_labels.iloc[:5000], scoring='neg_root_mean_squared_error', cv=3)
    pd.Series(selector_rmses).describe()
    return (selector_rmses,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Oh well, feature selection does not seem to help. But maybe that's just because the threshold we used was not optimal. Perhaps try tuning it using random search or grid search?
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Exercise: _Try creating a custom transformer that trains a k-Nearest Neighbors regressor (`sklearn.neighbors.KNeighborsRegressor`) in its `fit()` method, and outputs the model's predictions in its `transform()` method. Then add this feature to the preprocessing pipeline, using latitude and longitude as the inputs to this transformer. This will add a feature in the model that corresponds to the housing median price of the nearest districts._
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Rather than restrict ourselves to k-Nearest Neighbors regressors, let's create a transformer that accepts any regressor. For this, we can extend the `MetaEstimatorMixin` and have a required `estimator` argument in the constructor. The `fit()` method must work on a clone of this estimator, and it must also save `feature_names_in_`. The `MetaEstimatorMixin` will ensure that `estimator` is listed as a required parameters, and it will update `get_params()` and `set_params()` to make the estimator's hyperparameters available for tuning. Lastly, we create a `get_feature_names_out()` method: the output column name is the ...
        """
    )
    return


@app.cell
def _(BaseEstimator, TransformerMixin, check_is_fitted):
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.base import MetaEstimatorMixin, clone

    class FeatureFromRegressor(MetaEstimatorMixin, BaseEstimator, TransformerMixin):

        def __init__(self, estimator):
            self.estimator = estimator

        def fit(self, X, y=None):
            estimator_ = _clone(self.estimator)
            estimator_.fit(X, y)
            self.estimator_ = estimator_
            self.n_features_in_ = self.estimator_.n_features_in_
            if hasattr(self.estimator, 'feature_names_in_'):
                self.feature_names_in_ = self.estimator.feature_names_in_
            return self

        def transform(self, X):
            check_is_fitted(self)
            predictions = self.estimator_.predict(X)
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            return predictions

        def get_feature_names_out(self, names=None):
            check_is_fitted(self)
            n_outputs = getattr(self.estimator_, 'n_outputs_', 1)
            estimator_class_name = self.estimator_.__class__.__name__
            estimator_short_name = estimator_class_name.lower().replace('_', '')
            return [f'{estimator_short_name}_prediction_{i}' for i in range(n_outputs)]
    return FeatureFromRegressor, KNeighborsRegressor, MetaEstimatorMixin, clone


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's ensure it complies to Scikit-Learn's API:
        """
    )
    return


@app.cell
def _(FeatureFromRegressor, KNeighborsRegressor):
    from sklearn.utils.estimator_checks import check_estimator

    check_estimator(FeatureFromRegressor(KNeighborsRegressor()))
    return (check_estimator,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Good! Now let's test it:
        """
    )
    return


@app.cell
def _(FeatureFromRegressor, KNeighborsRegressor, housing_3, housing_labels):
    knn_reg = KNeighborsRegressor(n_neighbors=3, weights='distance')
    knn_transformer = FeatureFromRegressor(knn_reg)
    geo_features = housing_3[['latitude', 'longitude']]
    knn_transformer.fit_transform(geo_features, housing_labels)
    return geo_features, knn_reg, knn_transformer


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And what does its output feature name look like?
        """
    )
    return


@app.cell
def _(knn_transformer):
    knn_transformer.get_feature_names_out()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Okay, now let's include this transformer in our preprocessing pipeline:
        """
    )
    return


@app.cell
def _(ColumnTransformer, knn_transformer, preprocessing_2):
    from sklearn.base import clone
    transformers = [(name, _clone(transformer), columns) for name, transformer, columns in preprocessing_2.transformers]
    geo_index = [name for name, _, _ in transformers].index('geo')
    transformers[geo_index] = ('geo', knn_transformer, ['latitude', 'longitude'])
    new_geo_preprocessing = ColumnTransformer(transformers)
    return clone, geo_index, new_geo_preprocessing, transformers


@app.cell
def _(Pipeline, SVR, new_geo_preprocessing, rnd_search_1):
    new_geo_pipeline = Pipeline([('preprocessing', new_geo_preprocessing), ('svr', SVR(C=rnd_search_1.best_params_['svr__C'], gamma=rnd_search_1.best_params_['svr__gamma'], kernel=rnd_search_1.best_params_['svr__kernel']))])
    return (new_geo_pipeline,)


@app.cell
def _(cross_val_score, housing_3, housing_labels, new_geo_pipeline, pd):
    new_pipe_rmses = -cross_val_score(new_geo_pipeline, housing_3.iloc[:5000], housing_labels.iloc[:5000], scoring='neg_root_mean_squared_error', cv=3)
    pd.Series(new_pipe_rmses).describe()
    return (new_pipe_rmses,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Yikes, that's terrible! Apparently the cluster similarity features were much better. But perhaps we should tune the `KNeighborsRegressor`'s hyperparameters? That's what the next exercise is about.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Exercise: _Automatically explore some preparation options using `RandomSearchCV`._
        """
    )
    return


@app.cell
def _(
    RandomizedSearchCV,
    expon_1,
    housing_3,
    housing_labels,
    loguniform_1,
    new_geo_pipeline,
):
    _param_distribs = {'preprocessing__geo__estimator__n_neighbors': range(1, 30), 'preprocessing__geo__estimator__weights': ['distance', 'uniform'], 'svr__C': loguniform_1(20, 200000), 'svr__gamma': expon_1(scale=1.0)}
    new_geo_rnd_search = RandomizedSearchCV(new_geo_pipeline, param_distributions=_param_distribs, n_iter=50, cv=3, scoring='neg_root_mean_squared_error', random_state=42)
    new_geo_rnd_search.fit(housing_3.iloc[:5000], housing_labels.iloc[:5000])
    return (new_geo_rnd_search,)


@app.cell
def _(new_geo_rnd_search):
    new_geo_rnd_search_rmse = -new_geo_rnd_search.best_score_
    new_geo_rnd_search_rmse
    return (new_geo_rnd_search_rmse,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Oh well... at least we tried! It looks like the cluster similarity features are definitely better than the KNN feature. But perhaps you could try having both? And maybe training on the full training set would help as well.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 6.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Exercise: _Try to implement the `StandardScalerClone` class again from scratch, then add support for the `inverse_transform()` method: executing `scaler.inverse_transform(scaler.fit_transform(X))` should return an array very close to `X`. Then add support for feature names: set `feature_names_in_` in the `fit()` method if the input is a DataFrame. This attribute should be a NumPy array of column names. Lastly, implement the `get_feature_names_out()` method: it should have one optional `input_features=None` argument. If passed, the method should check that its length matches `n_features_in_`, and it should match `feature_names_in_` if it is defined, then `input_features` should be returned. If `input_features` is `None`, then the method should return `feature_names_in_` if it is defined or `np.array(["x0", "x1", ...])` with length `n_features_in_` otherwise._
        """
    )
    return


@app.cell
def _(BaseEstimator, TransformerMixin, check_array, check_is_fitted, np):
    class StandardScalerClone_1(BaseEstimator, TransformerMixin):

        def __init__(self, with_mean=True):
            self.with_mean = with_mean

        def fit(self, X, y=None):
            X_orig = X
            X = check_array(X)
            self.mean_ = X.mean(axis=0)
            self.scale_ = X.std(axis=0)
            self.n_features_in_ = X.shape[1]
            if hasattr(X_orig, 'columns'):
                self.feature_names_in_ = np.array(X_orig.columns, dtype=object)
            return self

        def transform(self, X):
            check_is_fitted(self)
            X = check_array(X)
            if self.n_features_in_ != X.shape[1]:
                raise ValueError('Unexpected number of features')
            if self.with_mean:
                X = X - self.mean_
            return X / self.scale_

        def inverse_transform(self, X):
            check_is_fitted(self)
            X = check_array(X)
            if self.n_features_in_ != X.shape[1]:
                raise ValueError('Unexpected number of features')
            X = X * self.scale_
            return X + self.mean_ if self.with_mean else X

        def get_feature_names_out(self, input_features=None):
            if input_features is None:
                return getattr(self, 'feature_names_in_', [f'x{i}' for i in range(self.n_features_in_)])
            else:
                if len(input_features) != self.n_features_in_:
                    raise ValueError('Invalid number of features')
                if hasattr(self, 'feature_names_in_') and (not np.all(self.feature_names_in_ == input_features)):
                    raise ValueError('input_features ≠ feature_names_in_')
                return input_features
    return (StandardScalerClone_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's test our custom transformer:
        """
    )
    return


@app.cell
def _(StandardScalerClone_1, check_estimator):
    check_estimator(StandardScalerClone_1())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        No errors, that's a great start, we respect the Scikit-Learn API.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's ensure the transformation works as expected:
        """
    )
    return


@app.cell
def _(StandardScalerClone_1, np):
    np.random.seed(42)
    X_1 = np.random.rand(1000, 3)
    scaler = StandardScalerClone_1()
    _X_scaled = scaler.fit_transform(X_1)
    assert np.allclose(_X_scaled, (X_1 - X_1.mean(axis=0)) / X_1.std(axis=0))
    return X_1, scaler


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        How about setting `with_mean=False`?
        """
    )
    return


@app.cell
def _(StandardScalerClone_1, X_1, np):
    scaler_1 = StandardScalerClone_1(with_mean=False)
    X_scaled_uncentered = scaler_1.fit_transform(X_1)
    assert np.allclose(X_scaled_uncentered, X_1 / X_1.std(axis=0))
    return X_scaled_uncentered, scaler_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And does the inverse work?
        """
    )
    return


@app.cell
def _(StandardScalerClone_1, X_1, np):
    scaler_2 = StandardScalerClone_1()
    X_back = scaler_2.inverse_transform(scaler_2.fit_transform(X_1))
    assert np.allclose(X_1, X_back)
    return X_back, scaler_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        How about the feature names out?
        """
    )
    return


@app.cell
def _(np, scaler_2):
    assert np.all(scaler_2.get_feature_names_out() == ['x0', 'x1', 'x2'])
    assert np.all(scaler_2.get_feature_names_out(['a', 'b', 'c']) == ['a', 'b', 'c'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        And if we fit a DataFrame, are the feature in and out ok?
        """
    )
    return


@app.cell
def _(StandardScalerClone_1, np, pd):
    df = pd.DataFrame({'a': np.random.rand(100), 'b': np.random.rand(100)})
    scaler_3 = StandardScalerClone_1()
    _X_scaled = scaler_3.fit_transform(df)
    assert np.all(scaler_3.feature_names_in_ == ['a', 'b'])
    assert np.all(scaler_3.get_feature_names_out() == ['a', 'b'])
    return df, scaler_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        All good! That's all for today! 😀
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Congratulations! You already know quite a lot about Machine Learning. :)
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()

