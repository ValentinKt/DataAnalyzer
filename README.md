# DataAnalyzer

DataAnalyzer is a Python library designed for statistical data analysis and automated test selection. 
It supports a range of statistical tests—from normality tests and t-tests to non-parametric, correlation, regression, and variance tests. 
Additionally, DataAnalyzer can be extended to use a machine learning model to reason about which test to run based on extracted data features.

# Basically

This is a tool to help you **select a statistical test**. In concrete terms, it is used **during the analysis or statistical inference** phase of your data, i.e. :

1. **After** you have collected, cleaned and explored the data (pre-processing).
2. **Before or in parallel** with any Deep Learning training, if you wish to test statistical hypotheses (e.g. comparing two groups, assessing the presence of a confounding variable, etc.).  

Typically, you use it **when you want to test the significance of differences** between your groups or variables (means, distribution, dependency) and need to determine the most suitable statistical procedure (parametric or non-parametric model, number of samples, paired or independent data, etc.).



## Features

- **Statistical Tests:**
  - **Normality:** Shapiro-Wilk test.
  - **Comparative Tests:** Student's t-test, Welch's t-test, Wilcoxon signed-rank test, Mann-Whitney U test.
  - **Correlation Tests:** Pearson and Spearman correlations.
  - **Regression:** Linear and non-linear regression.
  - **Variance Tests:** Levene's test.
  - **Multiple Group Comparison:** ANOVA and Kruskal-Wallis tests.
  - **Contingency Analysis:** Chi-square test.
  - **ANCOVA:** Analysis of covariance using a DataFrame.
- **Flexible Data Input:**
  - Accepts one or more datasets.
  - Validates data to ensure it meets the requirements for each test.
- **Model Reasoning:**
  - Optionally load a pre-trained model (e.g., a pickled scikit-learn estimator) that predicts the appropriate statistical test based on a rich set of extracted features.
  - Robust feature extraction including mean, standard deviation, skewness, kurtosis, sample size, and normality flag.
- **Centralized Logging:**
  - Detailed logging for debugging and tracking test execution.
- **Graphical User Interface (GUI):**
  - **Enhanced visualization** using Seaborn with custom colors, labels, and pointer control.
  - **Multi-selection support** for statistical tests and graphical representation.
  - **User-friendly design** for intuitive data analysis.

## Installation

To install the required dependencies, you can use `pip`:

```bash
pip install numpy pandas scipy statsmodels seaborn matplotlib
```

If you intend to use a machine learning model for test reasoning, make sure you have installed the dependencies for your model (for example, scikit-learn):

```bash
pip install scikit-learn
```

## Usage

Below is an example of how to use DataAnalyzer:

```python
import numpy as np
import pandas as pd
from data_analyzer import DataAnalyzer  # assuming the module is named data_analyzer.py

# Example datasets
data1 = [1, 2, 3, 4, 5]
data2 = [2, 4, 6, 8, 10]

# Initialize the analyzer with two datasets
analyzer = DataAnalyzer(data1, data2)

# Run a default comparison test (automatically selects t-test or non-parametric equivalent)
comparison_result = analyzer.run_test("comparison")
print("Comparison test result:", comparison_result)

# Run a chi-square test for independence
chi_result = analyzer.run_test("independence")
print("Chi-square result:", chi_result)

# Run a correlation test
correlation_result = analyzer.run_test("correlation")
print("Correlation test result:", correlation_result)

# Run linear regression
regression_result = analyzer.run_test("regression")
print("Linear regression result:", regression_result)

# Run Levene's test for equality of variances
levene_result = analyzer.run_test("variance")
print("Levene's test result:", levene_result)

# Run ANCOVA test with a DataFrame
df = pd.DataFrame({
    "y": [1, 2, 3, 4],
    "x": [1, 2, 3, 4],
    "cov": [2, 4, 6, 8]
})
analyzer_with_df = DataAnalyzer(data1, data2, df=df)
ancova_result = analyzer_with_df.run_test("ancova", formula="y ~ x + cov")
print("ANCOVA result:\n", ancova_result)

# Non-linear regression example
x_data = np.array([0, 1, 2, 3])
y_data = np.array([1, 2.7, 7.4, 20.1])
func = lambda x, a, b: a * np.exp(b * x)
analyzer_nl = DataAnalyzer(x_data, y_data)
non_linear_result = analyzer_nl.run_test("non_linear", func=func, p0=[1, 1])
print("Non-linear regression result:", non_linear_result)

# Multiple groups example (ANOVA or Kruskal-Wallis will be selected automatically)
group1 = [1, 2, 3, 4, 5]
group2 = [2, 3, 4, 5, 6]
group3 = [1.5, 2.5, 3.5, 4.5, 5.5]
analyzer_multi = DataAnalyzer(group1, group2, group3)
multi_result = analyzer_multi.run_test("comparison")
print("Multiple groups comparison test result:", multi_result)
```

## Model Reasoning

DataAnalyzer can optionally load a pre-trained model to reason about which statistical test to run based on the characteristics of the input data. The model is expected to implement a `predict` method and accept a feature vector that includes:

- Mean
- Standard deviation
- Skewness
- Kurtosis
- Sample size
- Normality flag

### Loading a Model

You can load a model from a file (e.g., a pickled scikit-learn model):

```python
analyzer.load_model("path_to_model.pkl")
```

### Using Model Reasoning

Once a model is loaded, you can use it to determine the test to run:

```python
model_test_result = analyzer.model_reason("comparison")
print("Model reasoning test result:", model_test_result)
```

The `model_reason` function has been refined to extract a robust feature set from the datasets, verify that the model is valid, and fall back to default behavior if any error occurs.

## Graphical User Interface (GUI)

The GUI enables users to easily select and analyze data using a visually appealing interface:

- **Multi-selection for statistical tests.**
- **Graph customization:** Colors, labels, and pointer control for enhanced clarity.
- **Seaborn integration** for high-quality visualizations.
- **User-friendly interface** for data input and visualization selection.

### Running the GUI

To launch the GUI, use:

```bash
seaborn gui_app.py
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for improvements and additional features.
