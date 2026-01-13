import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from matplotlib.patches import Rectangle
from scipy.stats import chi2_contingency
from pathlib import Path
from scipy.stats import pointbiserialr
from sklearn.feature_selection import mutual_info_classif


def plot_hazard_distribution(df: pd.DataFrame, save_dir="./output/visualizations"):
    """
    Plot the distribution of hazard types and save the figure.
    """

    # Ensure save directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot on the given axis
    df['hazard_type_name'].value_counts().plot(kind='bar', ax=ax)

    # Set labels and title
    ax.set_title("Hazard Type Distribution")
    ax.set_xlabel("Hazard Type")
    ax.set_ylabel("Count")

    # Adjust layout
    fig.tight_layout()

    # Save figure
    fig.savefig(save_dir / "hazard_type_distribution.png", dpi=300)

    # Close figure to free memory (optional if returning)
    plt.close(fig)

    return fig, ax


def plot_unknown_values(df: pd.DataFrame, save_dir="./output/visualizations"):
    """
    Plot the number of unknown (-1) values per feature and save the figure.
    """
    # Compute unknown counts
    unknown_counts = (df == -1).sum()
    unknown_counts = unknown_counts[unknown_counts > 0]

    if unknown_counts.empty:
        print("No unknown (-1) values found in the dataset.")
        return None

    # Ensure save directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot on axis
    unknown_counts.plot(kind='bar', ax=ax)

    # Labels and title
    ax.set_title("Unknown (-1) Value Counts per Feature")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Count")

    # Layout and save
    fig.tight_layout()
    fig.savefig(save_dir / "unknown_values_per_feature.png", dpi=300)

    # Close figure to free memory
    plt.close(fig)

    return fig, ax


def plot_tailight_vs_hazard(df: pd.DataFrame, save_dir="./output/visualizations"):
    # Ensure save directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Crosstab plot
    cross = pd.crosstab(df['tailight_status'], df['hazard_flag'])
    cross.plot(kind='bar', stacked=True, ax=ax)

    # Labels and title
    ax.set_title("Taillight Status vs Hazard")
    ax.set_xlabel("Taillight Status")
    ax.set_ylabel("Count")

    # Layout and save
    fig.tight_layout()
    fig.savefig(save_dir / "taillight_status_vs_hazard.png", dpi=300)

    # Close figure to free memory
    plt.close(fig)

    return fig, ax


def plot_visible_side_vs_hazard(df: pd.DataFrame, save_dir="./output/visualizations"):
    """
    Plot object visible side vs hazard flag and save the figure.
    """
    # Create crosstab
    cross = pd.crosstab(df['object_visible_side'], df['hazard_flag'])

    if cross.empty:
        print("No data found for object_visible_side vs hazard_flag.")
        return None

    # Ensure save directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot on the axis
    cross.plot(kind='bar', stacked=True, ax=ax)

    # Labels and title
    ax.set_title("Object Visible Side vs Hazard")
    ax.set_xlabel("Object Visible Side")
    ax.set_ylabel("Count")

    # Layout and save
    fig.tight_layout()
    fig.savefig(save_dir / "object_visible_side_vs_hazard.png", dpi=300)

    # Close figure to free memory
    plt.close(fig)

    return fig, ax


def plot_bbox_area(df: pd.DataFrame, save_dir="./output/visualizations"):
    """
    Plot the distribution of bounding box areas and save the figure.
    """
    # Ensure save directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    ax.hist(df['bbox_area'].dropna(), bins=50, color='skyblue', edgecolor='black')

    # Labels and title
    ax.set_title("Bounding Box Area Distribution")
    ax.set_xlabel("Area")
    ax.set_ylabel("Frequency")

    # Layout and save
    fig.tight_layout()
    fig.savefig(save_dir / "bbox_area_distribution.png", dpi=300)

    # Close figure to free memory
    plt.close(fig)

    return fig, ax


def plot_corr_among_numeric_features(df: pd.DataFrame, features, method="pearson", title=""):
    """
    Plot correlation matrix for numeric features and save figure.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    features : list[str]
        Numeric columns to compute correlation.
    method : str
        'pearson', 'spearman', or 'kendall'.
    title : str
        Optional title for the plot and filename.

    Returns
    -------
    corr : pd.DataFrame
        Correlation matrix.
    fig : matplotlib.figure.Figure
        Figure object.
    """
    # Compute correlation
    corr = df[features].corr(method=method)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        ax=ax,
        linewidths=0.5
    )

    # Title
    plot_title = f"{method.capitalize()} Correlation Matrix"
    if title:
        plot_title += f" ({title})"
    ax.set_title(plot_title)

    # Layout
    fig.tight_layout()

    # Ensure save directory exists
    save_dir = Path("./output/visualizations")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Safe filename
    safe_title = title.replace(" ", "_") if title else "correlation_matrix"
    filename = save_dir / f"{method.capitalize()}_Correlation_Matrix_{safe_title}.png"

    # Save figure
    fig.savefig(filename, dpi=300)

    # Close figure to free memory
    plt.close(fig)

    return fig, ax


def plot_corr_between_numeric_features_and_continuous_targets(df: pd.DataFrame, features: list, target: str, method="pearson", title=""):
    """
    Calculate and plot correlation of numeric features with target variable y.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    features : list[str]
        Numeric columns to compute correlation.
    target : str
        Target column name.
    method : str
        'pearson', 'spearman', or 'kendall'.
    title : str
        Optional title for the plot and filename.

    Returns
    -------
    corr_df : pd.DataFrame
        DataFrame of features and their correlation with target.
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    # Compute correlation of each feature with target
    corr_values = df[features + [target]].corr(method=method)[target].drop(target)
    corr_df = corr_values.reset_index()
    corr_df.columns = ['Feature', 'Correlation_with_y']
    corr_df = corr_df.sort_values(by='Correlation_with_y', key=abs, ascending=False)  # sort by absolute correlation

    # Plot as bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        x='Correlation_with_y',
        y='Feature',
        data=corr_df,
        palette="coolwarm",
        ax=ax
    )
    
    # Title
    plot_title = f"{method.capitalize()} Correlation with {target}"
    if title:
        plot_title += f" ({title})"
    ax.set_title(plot_title)
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Feature")

    # Layout
    fig.tight_layout()

    # Ensure save directory exists
    save_dir = Path("./output/visualizations")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Safe filename
    safe_title = title.replace(" ", "_") if title else "feature_correlation"
    filename = save_dir / f"{method.capitalize()}_Feature_Correlation_{safe_title}.png"

    # Save figure
    fig.savefig(filename, dpi=300)
    plt.close(fig)

    return fig, ax


def plot_corr_between_categorical_features_and_categorical_targets(
    df: pd.DataFrame,
    features,
    target="hazard_flag",
    method="chi2",
    title=""):
    """
    Compute and plot association between nominal categorical features and a target
    (binary or multi-class).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    features : list[str]
        Nominal categorical columns to analyze.
    target : str
        Target column (binary or multi-class).
    method : str
        'chi2' for Chi-square statistic, 'mutual_info' for Mutual Information.
    title : str
        Optional title for the plot and filename.

    Returns
    -------
    scores_df : pd.DataFrame
        DataFrame with features and association scores.
    fig : matplotlib.figure.Figure
        Figure object of the bar plot.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    scores = []

    if method == "chi2":
        for col in features:
            contingency_table = pd.crosstab(df[col], df[target])
            chi2_stat, p, dof, expected = chi2_contingency(contingency_table)
            
            # Warn if expected counts are too low
            if (expected < 5).any():
                print(f"Warning: Some expected counts < 5 for feature '{col}'")
            
            scores.append({"feature": col, "score": chi2_stat})
        score_label = "Chi-square"

    elif method == "mutual_info":
        # Encode categorical features and target
        X = df[features].apply(lambda x: x.astype("category").cat.codes)
        y = df[target].astype("category").cat.codes
        
        mi_scores = mutual_info_classif(X, y, discrete_features=True, random_state=42)
        for f, s in zip(features, mi_scores):
            scores.append({"feature": f, "score": s})
        score_label = "Mutual Information"

    else:
        raise ValueError("method must be 'chi2' or 'mutual_info'")

    # Convert scores to DataFrame
    scores_df = pd.DataFrame(scores).sort_values("score", ascending=False)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(x="score", y="feature", data=scores_df, palette="viridis", ax=ax)
    plot_title = f"{score_label} with Target"
    if title:
        plot_title += f" ({title})"
    ax.set_title(plot_title)
    ax.set_xlabel(score_label)
    ax.set_ylabel("Feature")

    fig.tight_layout()

    # Ensure save directory exists
    save_dir = Path("./output/visualizations")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Safe filename
    safe_title = title.replace(" ", "_") if title else "categorical_relation"
    filename = save_dir / f"{score_label}_with_Target_{safe_title}.png"
    fig.savefig(filename, dpi=300)
    plt.close(fig)

    return scores_df, fig, ax


def plot_corr_between_numeric_features_and_binary_target(df: pd.DataFrame, features, target="hazard_flag", save_dir="./output/visualizations", title="None"):
    """
    Compute point-biserial correlation between numeric features and a binary hazard target,
    plot the correlations as a horizontal bar chart, and save the figure.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    features : list[str]
        List of numeric feature columns.
    target : str
        Binary target column (default 'hazard_flag').
    save_dir : str
        Directory to save the figure.

    Returns
    -------
    results : pd.DataFrame
        DataFrame with correlation and p-value per feature.
    fig : matplotlib.figure.Figure
        Figure object.
    """
    results = []
    
    # Compute correlations
    for feature in features:
        valid = df[[feature, target]].dropna()
        corr, p_value = pointbiserialr(valid[target], valid[feature])
        results.append({"feature": feature, "correlation": corr, "p_value": p_value})
    
    results_df = pd.DataFrame(results).sort_values("correlation")

    if results_df.empty:
        print("No valid features for correlation.")
        return results_df, None

    # Ensure save directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot horizontal bar chart
    ax.barh(results_df["feature"], results_df["correlation"], color="skyblue", edgecolor="black")
    ax.axvline(0, color="black", linewidth=1)  # reference line

    ax.set_xlabel("Point-Biserial Correlation with Hazard")
    ax.set_ylabel("Feature")
    ax.set_title(f"Point-biserial Correlation with Hazard ({title})")

    fig.tight_layout()

    # Save figure
    filename = save_dir / "numeric_correlation_with_hazard.png"
    fig.savefig(filename, dpi=300)

    # Close figure to free memory
    plt.close(fig)

    return results_df, fig, ax


def cramers_v_plot(
    df: pd.DataFrame,
    features,
    target,
    save_dir="./output/visualizations",
    title=None):
    """
    Compute Cramér's V between categorical features and a target (binary or multi-class),
    plot the associations as a horizontal bar chart, and save the figure.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    features : list[str]
        Categorical feature column names.
    target : str
        Target column (binary or multi-class).
    save_dir : str
        Directory to save the figure.
    title : str
        Optional custom title for the plot.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with features and Cramér's V values.
    fig : matplotlib.figure.Figure
        Figure object of the bar plot.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    results = []

    for feature in features:
        # Drop missing values
        valid = df[[feature, target]].dropna()
        confusion = pd.crosstab(valid[feature], valid[target])

        if confusion.empty or min(confusion.shape) <= 1:
            v = np.nan
        else:
            chi2, _, _, _ = chi2_contingency(confusion)
            n = confusion.sum().sum()
            r, k = confusion.shape
            v = np.sqrt(chi2 / (n * (min(r, k) - 1)))

        results.append({"feature": feature, "cramers_v": v})

    results_df = pd.DataFrame(results).sort_values("cramers_v", ascending=False)

    if results_df.empty:
        print("No valid features for Cramér's V.")
        return results_df, None, None

    # Ensure save directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot horizontal bar chart
    ax.barh(results_df["feature"], results_df["cramers_v"], color="lightgreen", edgecolor="black")
    ax.set_xlabel("Cramér's V")
    ax.set_ylabel("Feature")

    plot_title = title if title else f"Categorical Feature Association with '{target}'"
    ax.set_title(plot_title)

    fig.tight_layout()

    # Save figure
    safe_title = plot_title.replace(" ", "_")
    filename = save_dir / f"categorical_correlation_{safe_title}.png"
    fig.savefig(filename, dpi=300)
    plt.close(fig)

    return results_df, fig, ax


def plot_feature_vs_hazard(df: pd.DataFrame, feature: str, target="hazard_flag", save_dir="./output/visualizations"):
    """
    Plot a numeric feature against the hazard flag using a boxplot,
    and save the figure as PDF.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    feature : str
        Numeric feature column name.
    target : str
        Binary target column name (default 'hazard_flag').
    save_dir : str
        Directory to save the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    # Ensure save directory exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot boxplot
    sns.boxplot(x=target, y=feature, data=df, ax=ax)

    # Labels and title
    ax.set_title(f"{feature} vs Hazard")
    ax.set_xlabel(target)
    ax.set_ylabel(feature)

    # Layout
    fig.tight_layout()

    # Safe filename
    safe_feature = feature.replace(" ", "_")
    filename = save_dir / f"{safe_feature}_vs_hazard.png"

    # Save figure
    fig.savefig(filename, dpi=300)

    # Close figure to free memory
    plt.close(fig)

    return fig, ax


"""
| Feature Type       | Target Type | Recommended Methods                                                 |
| ------------------ | ----------- | ------------------------------------------------------------------- |
| Numeric Continuous | Binary      | Point-Biserial, Mutual Information, Logistic Regression coefficient |
| Numeric Continuous | Multiclass  | ANOVA F-test, Kruskal-Wallis, Mutual Information                    |
| Categorical        | Binary      | Chi-Square, Cramér’s V, Mutual Information                          |
| Categorical        | Multiclass  | Chi-Square, Cramér’s V, Mutual Information                          |
"""

"""
| Metric                      | Question answered                                              | Scale                  |
| --------------------------- | -------------------------------------------------------------- | ---------------------- |
| **Chi-square (χ²)**         | *Is there statistical dependence?*                             | 0 → ∞ (not normalized) |
| **Cramér’s V**              | *How strong is the categorical association?*                   | 0 → 1                  |
| **Mutual Information (MI)** | *How much information does the feature give about the target?* | ≥ 0 (relative scale)   |

"""