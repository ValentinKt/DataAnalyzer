from typing import Tuple, Callable, Union, Optional, List, Any
import scipy.stats as stats
import logging
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import pickle
import numpy as np
from scipy import stats

# Centralized logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Type aliases for clarity
ArrayLike = Union[np.ndarray, List[Any], pd.Series]
TestResult = Union[Tuple[Any, ...], Any, None]

def _validate_array(data: ArrayLike, min_size: int = 1, ndim: int = 1) -> Optional[np.ndarray]:
    """Validate and convert input data to a numpy array."""
    try:
        arr = np.asarray(data)
        if arr.size < min_size:
            logger.warning(f"Data size {arr.size} is less than minimum required {min_size}.")
            return None
        if arr.ndim != ndim:
            logger.warning(f"Data has {arr.ndim} dimensions, expected {ndim}.")
            return None
        if np.ma.is_masked(arr) and np.ma.count_masked(arr) > 0:
            logger.warning("Masked arrays are not supported.")
            return None
        return arr
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid data: {e}")
        return None

def shapiro_wilk_test(data: ArrayLike) -> Tuple[Optional[float], Optional[float]]:
    """Perform the Shapiro-Wilk test for normality."""
    logger.info("Running Shapiro-Wilk test.")
    arr = _validate_array(data, min_size=3)
    if arr is None:
        return None, None
    try:
        statistic, p_value = stats.shapiro(arr)
        logger.info(f"Shapiro-Wilk: statistic={statistic:.4f}, p-value={p_value:.4f}")
        return statistic, p_value
    except Exception as e:
        logger.error(f"Shapiro-Wilk test failed: {e}")
        return None, None

def wilcoxon_signed_rank_test(data1: ArrayLike, data2: Optional[ArrayLike] = None) -> Tuple[Optional[float], Optional[float]]:
    """Perform the Wilcoxon signed-rank test."""
    logger.info("Running Wilcoxon signed-rank test.")
    arr1 = _validate_array(data1)
    if arr1 is None:
        return None, None
    if data2 is not None:
        arr2 = _validate_array(data2)
        if arr2 is None or arr1.size != arr2.size:
            logger.warning("Paired Wilcoxon requires equal-length datasets.")
            return None, None
        statistic, p_value = stats.wilcoxon(arr1, arr2)
    else:
        statistic, p_value = stats.wilcoxon(arr1)
    logger.info(f"Wilcoxon: statistic={statistic:.4f}, p-value={p_value:.4f}")
    return statistic, p_value

def t_test_student(data1: ArrayLike, data2: ArrayLike, equal_var: bool = True) -> Tuple[Optional[float], Optional[float]]:
    """Perform Student's t-test for independent samples."""
    logger.info("Running Student's t-test.")
    arr1, arr2 = _validate_array(data1), _validate_array(data2)
    if arr1 is None or arr2 is None:
        return None, None
    try:
        statistic, p_value = stats.ttest_ind(arr1, arr2, equal_var=equal_var)
        logger.info(f"Student's t-test: statistic={statistic:.4f}, p-value={p_value:.4f}")
        return statistic, p_value
    except Exception as e:
        logger.error(f"Student's t-test failed: {e}")
        return None, None

def t_test_welch(data1: ArrayLike, data2: ArrayLike) -> Tuple[Optional[float], Optional[float]]:
    """Perform Welch's t-test (unequal variances)."""
    logger.info("Running Welch's t-test.")
    arr1, arr2 = _validate_array(data1), _validate_array(data2)
    if arr1 is None or arr2 is None:
        return None, None
    try:
        statistic, p_value = stats.ttest_ind(arr1, arr2, equal_var=False)
        logger.info(f"Welch's t-test: statistic={statistic:.4f}, p-value={p_value:.4f}")
        return statistic, p_value
    except Exception as e:
        logger.error(f"Welch's t-test failed: {e}")
        return None, None

def mann_whitney_u_test(data1: ArrayLike, data2: ArrayLike) -> Tuple[Optional[float], Optional[float]]:
    """Perform the Mann-Whitney U test."""
    logger.info("Running Mann-Whitney U test.")
    arr1, arr2 = _validate_array(data1), _validate_array(data2)
    if arr1 is None or arr2 is None:
        return None, None
    try:
        statistic, p_value = stats.mannwhitneyu(arr1, arr2)
        logger.info(f"Mann-Whitney U: statistic={statistic:.4f}, p-value={p_value:.4f}")
        return statistic, p_value
    except Exception as e:
        logger.error(f"Mann-Whitney U test failed: {e}")
        return None, None

def chi_square_test(observed: ArrayLike) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[np.ndarray]]:
    """Perform the Chi-square test for independence."""
    logger.info("Running Chi-square test.")
    arr = _validate_array(observed, min_size=1, ndim=2)
    if arr is None or np.any(arr < 0):
        logger.warning("Chi-square test requires a 2D contingency table with non-negative values.")
        return None, None, None, None
    try:
        chi2, p_value, dof, expected = stats.chi2_contingency(arr)
        logger.info(f"Chi-square: chi2={chi2:.4f}, p-value={p_value:.4f}, dof={dof}")
        return chi2, p_value, dof, expected
    except Exception as e:
        logger.error(f"Chi-square test failed: {e}")
        return None, None, None, None

def pearson_correlation_test(data1: ArrayLike, data2: ArrayLike) -> Tuple[Optional[float], Optional[float]]:
    """Perform Pearson's correlation test."""
    logger.info("Running Pearson's correlation test.")
    arr1, arr2 = _validate_array(data1), _validate_array(data2)
    if arr1 is None or arr2 is None or arr1.size != arr2.size:
        logger.warning("Pearson correlation requires equal-length datasets.")
        return None, None
    try:
        correlation, p_value = stats.pearsonr(arr1, arr2)
        logger.info(f"Pearson: correlation={correlation:.4f}, p-value={p_value:.4f}")
        return correlation, p_value
    except Exception as e:
        logger.error(f"Pearson correlation test failed: {e}")
        return None, None

def spearman_correlation_test(data1: ArrayLike, data2: ArrayLike) -> Tuple[Optional[float], Optional[float]]:
    """Perform Spearman's correlation test."""
    logger.info("Running Spearman's correlation test.")
    arr1, arr2 = _validate_array(data1), _validate_array(data2)
    if arr1 is None or arr2 is None or arr1.size != arr2.size:
        logger.warning("Spearman correlation requires equal-length datasets.")
        return None, None
    try:
        correlation, p_value = stats.spearmanr(arr1, arr2)
        logger.info(f"Spearman: correlation={correlation:.4f}, p-value={p_value:.4f}")
        return correlation, p_value
    except Exception as e:
        logger.error(f"Spearman correlation test failed: {e}")
        return None, None

def kruskal_wallis_test(*args: ArrayLike) -> Tuple[Optional[float], Optional[float]]:
    """Perform the Kruskal-Wallis test."""
    logger.info("Running Kruskal-Wallis test.")
    if len(args) < 2:
        logger.warning("Kruskal-Wallis requires at least two groups.")
        return None, None
    arrays = [_validate_array(arg) for arg in args]
    if any(arr is None for arr in arrays):
        return None, None
    try:
        statistic, p_value = stats.kruskal(*arrays)
        logger.info(f"Kruskal-Wallis: statistic={statistic:.4f}, p-value={p_value:.4f}")
        return statistic, p_value
    except Exception as e:
        logger.error(f"Kruskal-Wallis test failed: {e}")
        return None, None

def anova_one_way(*args: ArrayLike) -> Tuple[Optional[float], Optional[float]]:
    """Perform one-way ANOVA."""
    logger.info("Running one-way ANOVA.")
    if len(args) < 2:
        logger.warning("ANOVA requires at least two groups.")
        return None, None
    arrays = [_validate_array(arg) for arg in args]
    if any(arr is None for arr in arrays):
        return None, None
    try:
        f_statistic, p_value = stats.f_oneway(*arrays)
        logger.info(f"ANOVA: F={f_statistic:.4f}, p-value={p_value:.4f}")
        return f_statistic, p_value
    except Exception as e:
        logger.error(f"ANOVA failed: {e}")
        return None, None

def ancova_test(data: pd.DataFrame, formula: str) -> Optional[pd.DataFrame]:
    """Perform ANCOVA test using statsmodels."""
    logger.info(f"Running ANCOVA with formula: {formula}")
    if not isinstance(data, pd.DataFrame):
        logger.error("ANCOVA requires a pandas DataFrame.")
        return None
    if data.shape[0] == 0:
        logger.warning("ANCOVA requires non-empty data.")
        return None
    try:
        model = smf.ols(formula, data=data).fit()
        anova_table = smf.anova_lm(model)
        logger.info(f"ANCOVA results:\n{anova_table}")
        return anova_table
    except Exception as e:
        logger.error(f"ANCOVA test failed: {e}")
        return None

def linear_regression_test(data1: ArrayLike, data2: ArrayLike) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Perform linear regression."""
    logger.info("Running linear regression.")
    arr1, arr2 = _validate_array(data1), _validate_array(data2)
    if arr1 is None or arr2 is None or arr1.size != arr2.size:
        logger.warning("Linear regression requires equal-length datasets.")
        return None, None, None, None, None
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(arr1, arr2)
        logger.info(f"Linear regression: slope={slope:.4f}, intercept={intercept:.4f}, r={r_value:.4f}, p-value={p_value:.4f}, std_err={std_err:.4f}")
        return slope, intercept, r_value, p_value, std_err
    except Exception as e:
        logger.error(f"Linear regression failed: {e}")
        return None, None, None, None, None

def non_linear_regression_test(x_data: ArrayLike, y_data: ArrayLike, func: Callable, p0: Optional[ArrayLike] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Perform non-linear regression."""
    logger.info("Running non-linear regression.")
    x_arr, y_arr = _validate_array(x_data), _validate_array(y_data)
    if x_arr is None or y_arr is None or x_arr.size != y_arr.size:
        logger.warning("Non-linear regression requires equal-length datasets.")
        return None, None
    if not callable(func):
        logger.error("Function must be callable.")
        return None, None
    try:
        if p0 is None:
            n_params = func.__code__.co_argcount - 1
            p0 = np.ones(n_params)
        popt, pcov = curve_fit(func, x_arr, y_arr, p0=p0)
        logger.info(f"Non-linear regression: parameters={popt}")
        return popt, pcov
    except (RuntimeError, ValueError) as e:
        logger.error(f"Non-linear regression failed: {e}")
        return None, None

def levene_test(data1: ArrayLike, *args: ArrayLike) -> Tuple[Optional[float], Optional[float]]:
    """Perform Levene's test for equality of variances."""
    logger.info("Running Levene's test.")
    arr1 = _validate_array(data1)
    if arr1 is None:
        return None, None
    arrays = [arr1] + [_validate_array(arg) for arg in args]
    if any(arr is None for arr in arrays):
        logger.warning("Levene's test requires all datasets to be valid.")
        return None, None
    try:
        statistic, p_value = stats.levene(*arrays)
        logger.info(f"Levene's: statistic={statistic:.4f}, p-value={p_value:.4f}")
        return statistic, p_value
    except Exception as e:
        logger.error(f"Levene's test failed: {e}")
        return None, None

class DataAnalyzer:
    """Class for analyzing data and selecting appropriate statistical tests."""
    def __init__(self, *data: ArrayLike, df: Optional[pd.DataFrame] = None, model: Optional[Any] = None):
        """Initialize the DataAnalyzer with one or more datasets, an optional DataFrame, and an optional model."""
        logger.info("Initializing DataAnalyzer.")
        if not data:
            raise ValueError("At least one dataset must be provided.")
        self.data = []
        for d in data:
            arr = _validate_array(d)
            if arr is None:
                raise ValueError("Invalid dataset provided.")
            self.data.append(arr)
        self.df = df if isinstance(df, pd.DataFrame) else None
        self.nb_groups = len(self.data)
        self.normality = [self.is_normally_distributed(arr) for arr in self.data]
        self.model = model  # Optionally pass a model during initialization.
        logger.info(f"Initialized with {self.nb_groups} group(s).")

    def is_normally_distributed(self, data: ArrayLike) -> bool:
        """Check if data is normally distributed using Shapiro-Wilk test."""
        statistic, p_value = shapiro_wilk_test(data)
        return p_value > 0.05 if p_value is not None else True

    def are_equal_lengths(self) -> bool:
        """Check if all datasets have equal lengths."""
        return all(arr.size == self.data[0].size for arr in self.data)

    def load_model(self, model_path: str) -> None:
        """Load a model (e.g., a pickled model) from the given file path."""
        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            logger.info(f"Model loaded successfully from {model_path}.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def select_test(self, test_type: str = "comparison", **kwargs) -> Tuple[str, Optional[Callable]]:
        """Select an appropriate statistical test based on data characteristics."""
        logger.info(f"Selecting test for {test_type}.")
        # For one dataset
        if self.nb_groups == 1:
            if test_type == "comparison":
                return "Shapiro-Wilk", shapiro_wilk_test
            else:
                return "Unknown", None
        # For exactly two datasets
        elif self.nb_groups == 2:
            if test_type == "comparison":
                if self.normality[0] and self.normality[1]:
                    if self.are_equal_lengths():
                        return "Paired t-test", t_test_student
                    else:
                        return "Welch's t-test", t_test_welch
                else:
                    if self.are_equal_lengths():
                        return "Wilcoxon Signed-Rank", wilcoxon_signed_rank_test
                    else:
                        return "Mann-Whitney U", mann_whitney_u_test
            elif test_type == "correlation":
                if self.normality[0] and self.normality[1]:
                    return "Pearson Correlation", pearson_correlation_test
                else:
                    return "Spearman Correlation", spearman_correlation_test
            elif test_type == "regression":
                return "Linear Regression", linear_regression_test
            elif test_type == "non_linear":
                return "Non-linear Regression", non_linear_regression_test
            elif test_type == "ancova":
                if self.df is None:
                    logger.warning("ANCOVA requires a DataFrame (df parameter).")
                    return "Unknown", None
                return "ANCOVA", ancova_test
            elif test_type == "independence":
                return "Chi-square", chi_square_test
            elif test_type == "variance":
                return "Levene's Test", levene_test
            else:
                return "Unknown", None
        # For more than two datasets (multiple groups)
        else:
            if test_type == "comparison":
                # Use ANOVA if all groups are normal; otherwise, use Kruskal-Wallis.
                if all(self.normality):
                    return "ANOVA", anova_one_way
                else:
                    return "Kruskal-Wallis", kruskal_wallis_test
            elif test_type in ("kruskal", "anova"):
                if test_type == "kruskal":
                    return "Kruskal-Wallis", kruskal_wallis_test
                else:
                    return "ANOVA", anova_one_way
            elif test_type == "variance":
                return "Levene's Test", levene_test
            else:
                return "Unknown", None

    def run_test(self, test_type: str = "comparison", **kwargs) -> TestResult:
        """Run the selected statistical test."""
        test_name, test_func = self.select_test(test_type, **kwargs)
        if test_func is None:
            logger.warning(f"No suitable test found: {test_name}")
            return None
        logger.info(f"Running {test_name}.")
        try:
            if test_name == "ANCOVA":
                if "formula" not in kwargs:
                    raise ValueError("ANCOVA requires a 'formula' argument.")
                return test_func(self.df, kwargs["formula"])
            elif test_name == "Non-linear Regression":
                if "func" not in kwargs or not callable(kwargs["func"]):
                    raise ValueError("Non-linear regression requires a callable 'func' argument.")
                if self.nb_groups != 2:
                    raise ValueError("Non-linear regression requires exactly two datasets.")
                return test_func(self.data[0], self.data[1], kwargs["func"], kwargs.get("p0"))
            elif test_name == "Chi-square":
                if self.nb_groups != 2:
                    raise ValueError("Chi-square test requires exactly two datasets to form a contingency table.")
                return test_func(np.array([self.data[0], self.data[1]]))
            elif self.nb_groups == 1:
                return test_func(self.data[0])
            else:
                return test_func(*self.data)
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return None

def model_reason(self, default_test_type: str = "comparison", **kwargs) -> TestResult:
    """
    Use the loaded model to reason about which test to run.
    
    This enhanced version extracts a robust set of features from each dataset:
      - Mean
      - Standard deviation
      - Skewness
      - Kurtosis
      - Sample size
      - Normality flag (1 if normally distributed, 0 otherwise)
    
    These features are concatenated into a single feature vector (flattened as needed)
    and passed to the model's `predict` method. The model is expected to return a suggested 
    test type which is then used to run the corresponding statistical test.
    
    If no model is loaded or if any error occurs, the function falls back to the default
    test selection and execution.
    """
    # Check if a model is loaded and has a predict method.
    if self.model is None or not hasattr(self.model, "predict") or not callable(self.model.predict):
        logger.info("No valid model loaded. Falling back to default test selection.")
        return self.run_test(default_test_type, **kwargs)
    
    try:
        # Build feature set for each dataset.
        features = []
        for arr in self.data:
            arr = np.asarray(arr)
            # Calculate a set of descriptive statistics.
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            skew_val = stats.skew(arr)
            kurtosis_val = stats.kurtosis(arr)
            sample_size = arr.size
            normal_flag = 1 if self.is_normally_distributed(arr) else 0

            features.extend([mean_val, std_val, skew_val, kurtosis_val, sample_size, normal_flag])
        
        # Reshape into 2D array (one sample with multiple features).
        feature_vector = np.array(features).reshape(1, -1)
        logger.info(f"Extracted features for model reasoning: {feature_vector}")
        
        # Predict the test type using the loaded model.
        predicted_test = self.model.predict(feature_vector)
        if isinstance(predicted_test, (list, np.ndarray)):
            predicted_test = predicted_test[0]
        logger.info(f"Model predicted test: {predicted_test}")
        
        # Run the test indicated by the model.
        return self.run_test(predicted_test, **kwargs)
    except Exception as e:
        logger.error(f"Enhanced model reasoning failed: {e}. Falling back to default test selection.")
        return self.run_test(default_test_type, **kwargs)

# Example usage
if __name__ == "__main__":
    # Basic comparison test with two datasets
    data1 = [1, 2, 3, 4, 5]
    data2 = [2, 4, 6, 8, 10]
    analyzer = DataAnalyzer(data1, data2)
    result = analyzer.run_test("comparison")
    print(f"Comparison test result: {result}")

    # Load a model (for example purposes, the model file should be a pickled object with a predict method)
    # analyzer.load_model("path_to_model.pkl")

    # Use model reasoning if a model is loaded (otherwise, it falls back to default)
    model_result = analyzer.model_reason("comparison")
    print(f"Model reasoning test result: {model_result}")

    # Chi-square test (requires exactly two datasets forming a contingency table)
    chi_result = analyzer.run_test("independence")
    print(f"Chi-square result: {chi_result}")

    # Spearman correlation
    spearman_result = analyzer.run_test("correlation")
    print(f"Spearman correlation result: {spearman_result}")

    # Linear regression
    lin_reg_result = analyzer.run_test("regression")
    print(f"Linear regression result: {lin_reg_result}")

    # Levene's test for equality of variances with two datasets
    levene_result = analyzer.run_test("variance")
    print(f"Levene's test result: {levene_result}")

    # ANCOVA test with a DataFrame
    df = pd.DataFrame({"y": [1, 2, 3, 4], "x": [1, 2, 3, 4], "cov": [2, 4, 6, 8]})
    analyzer_with_df = DataAnalyzer(data1, data2, df=df)
    ancova_result = analyzer_with_df.run_test("ancova", formula="y ~ x + cov")
    print(f"ANCOVA result:\n{ancova_result}")

    # Non-linear regression (using two datasets)
    x_data = np.array([0, 1, 2, 3])
    y_data = np.array([1, 2.7, 7.4, 20.1])
    func = lambda x, a, b: a * np.exp(b * x)
    analyzer_nl = DataAnalyzer(x_data, y_data)
    nl_result = analyzer_nl.run_test("non_linear", func=func, p0=[1, 1])
    print(f"Non-linear regression result: {nl_result}")

    # Comparison test with more than two groups
    group1 = [1, 2, 3, 4, 5]
    group2 = [2, 3, 4, 5, 6]
    group3 = [1.5, 2.5, 3.5, 4.5, 5.5]
    analyzer_multi = DataAnalyzer(group1, group2, group3)
    multi_result = analyzer_multi.run_test("comparison")
    print(f"Multiple groups comparison test result: {multi_result}")
