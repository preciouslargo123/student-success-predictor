import pandas as pd

RAW_FEATURES = [
    "Submission.Actions", "Quiz.Actions", "Forum.Actions", "Create.Actions",
    "Read.Actions", "Update.Actions", "Delete.Actions", "Days.Logged.In",
]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add activity-per-day rates, total activity and share of reading."""
    X = X[RAW_FEATURES].copy()
    days = X["Days.Logged.In"] + 1
    for c in ["Submission.Actions", "Quiz.Actions", "Forum.Actions",
              "Create.Actions", "Read.Actions", "Update.Actions"]:
        X[c + "_perday"] = X[c] / days
    X["total_actions"] = X[RAW_FEATURES[:7]].sum(axis=1)
    X["read_share"] = X["Read.Actions"] / (X["total_actions"] + 1)
    return X