import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from data_analyzer import DataAnalyzer  # Ensure your DataAnalyzer module is available

# --- Sidebar ---
st.sidebar.title("DataAnalyzer Options")

# File uploader for CSV file
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# Multi-select for statistical tests.
test_options = [
    "comparison", "correlation", "regression", "non_linear",
    "ancova", "independence", "variance", "kruskal", "anova"
]
selected_tests = st.sidebar.multiselect("Select Statistical Tests", test_options, default=["comparison"])

# Multi-select for graph types.
graph_options = st.sidebar.multiselect(
    "Select Graph Types",
    ["Histogram", "KDE", "Boxplot", "Countplot", "Pairplot"],
    default=["Histogram", "Boxplot"]
)

# --- Main App ---
st.title("DataAnalyzer GUI")
st.write(
    "Welcome to the DataAnalyzer GUI. Upload your dataset, select columns for analysis, "
    "and run statistical tests. This app now supports all data types with interactive graphs."
)

if uploaded_file is not None:
    try:
        # Read CSV into DataFrame.
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # Let the user select columns for analysis.
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select Columns for Analysis", all_columns, default=all_columns)

        if len(selected_columns) < 1:
            st.error("Please select at least one column for analysis.")
        else:
            data = []  # Will store numeric arrays for analysis.
            st.subheader("Visualizations")
            for col in selected_columns:
                col_data = df[col].dropna()
                st.markdown(f"### Column: {col}")

                # Numeric Columns
                if pd.api.types.is_numeric_dtype(col_data):
                    data.append(col_data.values)
                    if "Histogram" in graph_options:
                        fig = px.histogram(
                            col_data,
                            nbins=30,
                            title=f"Histogram of {col}",
                            labels={"value": col},
                            color_discrete_sequence=["#636EFA"]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    if "KDE" in graph_options:
                        # Using histogram with marginal violin to mimic density.
                        fig = px.histogram(
                            col_data,
                            nbins=30,
                            marginal="violin",
                            title=f"Density & KDE of {col}",
                            labels={"value": col},
                            color_discrete_sequence=["#EF553B"]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    if "Boxplot" in graph_options:
                        fig = px.box(
                            col_data,
                            points="all",
                            title=f"Boxplot of {col}",
                            labels={"value": col},
                            color_discrete_sequence=["#00CC96"]
                        )
                        st.plotly_chart(fig, use_container_width=True)
                # Datetime Columns
                elif pd.api.types.is_datetime64_any_dtype(col_data):
                    col_data_dt = pd.to_datetime(col_data, errors="coerce").dropna()
                    if col_data_dt.empty:
                        st.warning(f"Column '{col}' could not be parsed as dates.")
                    else:
                        # Convert to Unix timestamps for analysis.
                        timestamps = col_data_dt.astype(np.int64) // 10**9
                        data.append(timestamps.values)
                        fig = px.line(
                            x=col_data_dt,
                            y=np.arange(len(col_data_dt)),
                            title=f"Time Series of {col}",
                            labels={"x": col, "y": "Index"},
                            markers=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                # Categorical or other Types
                else:
                    # Factorize categorical data to numeric codes.
                    codes, uniques = pd.factorize(col_data)
                    data.append(codes)
                    fig = px.histogram(
                        col_data,
                        title=f"Countplot of {col}",
                        labels={col: "Count"},
                        color_discrete_sequence=["#AB63FA"]
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Show a pairplot (scatter matrix) if more than one column is selected.
            if len(selected_columns) > 1 and "Pairplot" in graph_options:
                st.subheader("Pairplot")
                try:
                    fig = px.scatter_matrix(
                        df[selected_columns],
                        title="Scatter Matrix",
                        color=selected_columns[0],  # Color by the first selected column
                        height=700,
                        width=700
                    )
                    fig.update_traces(diagonal_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to create pairplot: {e}")

            # Run statistical tests.
            st.subheader("Test Results")
            for test in selected_tests:
                st.markdown(f"#### {test.capitalize()} Test")
                try:
                    # For ANCOVA tests, pass the full DataFrame.
                    analyzer = DataAnalyzer(*data, df=df if test == "ancova" else None)
                    result = analyzer.run_test(test)
                    st.write(result)
                except Exception as e:
                    st.error(f"Error running {test} test: {e}")
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
else:
    st.info("Please upload a CSV file from the sidebar to get started.")
