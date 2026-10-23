from pandas import DataFrame, Series


def prepare_shap_data(df: DataFrame) -> tuple[DataFrame, Series, dict, list]:
    """
    Returns:
        test_x (pd.DataFrame): Feature data for testing.
        test_y (pd.Series): Labels for testing.
        column_groups (dict): Dictionary with column group names and column lists.
    """
    # Intermediate tables omit s_dim_* entirely when no side channel is present.
    excluded_columns = {"datapoint_idx", "labels"}
    side_columns = [column for column in df.columns if column.startswith("s_dim_")]
    concept_columns = [
        column
        for column in df.columns
        if column not in excluded_columns and column not in side_columns
    ]
    if "labels" not in df.columns:
        raise ValueError("SAGE input dataframe must contain a 'labels' column")
    if not concept_columns:
        raise ValueError("SAGE input dataframe contains no concept features")

    column_groups_names = {"concepts": concept_columns}
    if side_columns:
        column_groups_names["side_channel"] = side_columns

    feature_columns = concept_columns + side_columns
    df_x = df[feature_columns]
    df_y = df["labels"]

    cols = df_x.columns.tolist()

    # Group indices
    groups = []
    for _, group in column_groups_names.items():
        ind_list = []
        for feature in group:
            ind_list.append(cols.index(feature))
        groups.append(ind_list)

    return df_x, df_y, column_groups_names, groups


def group_importance_metric(explanation_values: dict[str, float]) -> float:
    if "concepts" not in explanation_values:
        raise ValueError("SAGE explanation is missing the concepts group")
    if "side_channel" not in explanation_values:
        return 1.0
    total_sum = sum(explanation_values.values())
    return explanation_values["concepts"] / total_sum if total_sum else 0.0
