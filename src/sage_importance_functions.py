from pandas import DataFrame, Series


def prepare_shap_data(df: DataFrame) -> tuple[DataFrame, Series, dict, list]:
    """
    Returns:
        test_x (pd.DataFrame): Feature data for testing.
        test_y (pd.Series): Labels for testing.
        column_groups (dict): Dictionary with column group names and column lists.
    """
    import numpy as np

    # Step 2: Identify column groups
    all_columns = list(df.columns)
    label_idx = all_columns.index("labels")
    s_dim_start_idx = all_columns.index("s_dim_0")

    # Group 1: Columns between "label" and "s_dim_0"
    group1_columns = all_columns[label_idx + 1 : s_dim_start_idx]
    # Group 2: Columns from "s_dim_0" to the end
    group2_columns = all_columns[s_dim_start_idx:]

    column_groups_names = {"concepts": group1_columns, "side_channel": group2_columns}

    # Step 4: Extract test features and labels
    df_x = df[group1_columns + group2_columns]
    df_y = df["labels"]

    ### From sage airbnb.ipynb example:
    feature_names = group1_columns + group2_columns
    group_names = [group for group in column_groups_names]
    cols = df_x.columns.tolist()
    for col in feature_names:
        if np.all([col not in group[1] for group in column_groups_names.items()]):
            group_names.append(col)

    # Group indices
    groups = []
    for _, group in column_groups_names.items():
        ind_list = []
        for feature in group:
            ind_list.append(cols.index(feature))
        groups.append(ind_list)

    return df_x, df_y, column_groups_names, groups


def group_importance_metric(explanation_values: dict[str, float]) -> float:
    if len(explanation_values) != 2:
        raise ValueError(
            "More than just concepts and side channel. Maybe there is a mistake somewhere?"
        )
    total_sum = sum(explanation_values.values())
    return explanation_values["concepts"] / total_sum
