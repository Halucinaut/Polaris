"""Pure utility functions for evaluation scripts — no MLX dependency."""


def select_eval_datasets(
    test_data: str,
    review_data: str,
    skip_review: bool,
) -> list[tuple[str, str]]:
    """Return the list of (label, path_str) pairs to evaluate.

    Pure function with no side effects — safe to test without MLX.
    """
    datasets = [("test", test_data)]
    if not skip_review:
        datasets.append(("review", review_data))
    return datasets
