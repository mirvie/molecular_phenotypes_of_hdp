from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import rgb2hex
from sklearn.metrics import confusion_matrix, log_loss, roc_auc_score, roc_curve


def get_binary_classifier_performance_stats(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    prob_thres: float | None = None,
) -> dict[str, float | np.ndarray]:
    """
    Calculate key stats & performance metrics for a binary classifier predicting outcome
    y_true with predictions y_pred for a single set. These metrics are:

        * Statistics, such as number of cases, controls, & prevalence in this set
        * auc, fpr (raw & interpolated), tpr (raw & interpolated) if y_pred is a
          probability w/intermediate probabilities
        * log loss if y_pred is a probability w/intermediate probabilities
        * sensitivity
        * specificity
        * Positive likelihood ratio (LR_plus)
        * Negative Likelihood ratio (LR_minus)
        * Diagnostic Odds Ratio (DOR)
        * PPV
        * NPV
        * positive rate

    Args:
        y_true: Sample outcomes. Values can be True/False or 1/0 for case/control,
            resp., for each sample
        y_pred: Sample probabilities or class predictions  corresponding to
            y_true.  If class predictions, must have values that match the set of
            values in y_true.
        prob_thres: If y_pred are probabilities, threshold needs to be given to
            convert to classification. Value is None when y_pred are class predictions.
            Default None.

    Returns:
        Dictionary keyed by metrics, including medians and 95% CI estimates by bootstrap


    Returns:
        Dictionary where keys are metric names and values are metric values.

    """
    if isinstance(y_true, np.ndarray):
        y_true = pd.Series(y_true)
    if isinstance(y_pred, np.ndarray):
        y_pred = pd.Series(y_pred)
    assert not (y_true.isna().any()), "NAs found in y_true"
    assert not (y_pred.isna().any()), "NAs found in y_pred"
    assert len(y_true) == len(y_pred), "Unequal lengths of y_true and y_pred"
    assert set(y_true) == {
        True,
        False,
    }, "Two true classes, either (True, False) or (1,0), required"
    y_true = y_true.astype(int)
    # assert y_pred is boolean or numeric
    assert pd.api.types.is_bool_dtype(y_pred) or pd.api.types.is_numeric_dtype(
        y_pred
    ), (
        "y_pred must be a classification (True, False) or (1,0), or a numeric "
        "probability between 0 and 1"
    )
    pred_is_prob = not set(y_pred).issubset({True, False})
    if pred_is_prob:
        assert ((y_pred <= 1).all()) & ((y_pred >= 0).all()), (
            "y_pred must be a classification (True, False) or (1,0), or a numeric "
            "probability between 0 and 1"
        )
    else:
        assert (
            prob_thres is None
        ), "Cannot apply probability thresholds to class predictions"

    n_cases = (y_true == 1).sum()
    n_controls = (y_true == 0).sum()
    results: dict[str, float | np.ndarray] = {
        "n_cases": n_cases,
        "n_controls": n_controls,
        "prevalence": n_cases / (n_cases + n_controls) * 100,
    }
    if pred_is_prob:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        fpr_interp = np.linspace(0, 1, 101)
        tpr_interp = np.interp(fpr_interp, fpr, tpr)
        results = {
            **results,
            "AUC": roc_auc_score(y_true, y_pred, labels=[0, 1]),
            "log_loss": log_loss(y_true, y_pred, labels=[0, 1]),
            "fpr": fpr,
            "tpr": tpr,
            "fpr_interp": fpr_interp,
            "tpr_interp": tpr_interp,
        }
        y_pred_class = y_pred >= prob_thres
    else:
        results = {
            **results,
            "AUC": np.nan,
            "log_loss": np.nan,
            "fpr": np.nan,
            "tpr": np.nan,
            "fpr_interp": np.nan,
            "tpr_interp": np.nan,
        }
        y_pred_class = y_pred
    tn, fp, fn, tp = confusion_matrix(
        y_true=y_true,
        y_pred=y_pred_class.astype(int),
        labels=[0, 1],
    ).ravel()
    # numpy to handle divide by 0/inf
    with np.errstate(
        divide="ignore", invalid="ignore"
    ):  # silence 0/0 warning resulting in np.nan and invalid values for scalar divide
        sensitivity = np.int64(tp) / (fn + tp)
        specificity = np.int64(tn) / (fp + tn)
        ppv = np.int64(tp) / (tp + fp)
        npv = np.int64(tn) / (tn + fn)

        if np.isnan(sensitivity) or np.isnan(specificity):
            lr_plus = np.nan
            lr_minus = np.nan
        else:
            if specificity == 1:
                lr_plus = np.inf
            elif np.isinf(specificity):
                lr_plus = 0
            else:
                lr_plus = sensitivity / (1 - specificity)

            if specificity == 0:
                lr_minus = np.inf
            elif np.isinf(specificity):
                lr_minus = 0
            else:
                lr_minus = (1 - sensitivity) / specificity

    results = {
        **results,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "lr_plus": lr_plus,
        "lr_minus": lr_minus,
        "DOR": lr_plus / lr_minus,
        "PPV": ppv,
        "NPV": npv,
        "positive_rate": np.int64(tp + fp) / (tn + fp + fn + tp),
    }

    return results


def get_binary_classifier_performance_stats_with_ci(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    prob_thres: float | None = None,
    n_boot_iters: int | None = 10_000,
    alpha_bootstrap: float = 0.05,
) -> dict:
    """
    Calculate key stats & performance metrics for a binary classifier predicting outcome
    y_true with predictions y_pred for the actual samples and add bootstrap statistics
    such as the 95% CI. These metrics are:

        * Statistics, such as number of cases, controls, & prevalence in this set
        * auc, fpr (raw & interpolated), tpr (raw & interpolated) if y_pred is a
          probability w/intermediate probabilities
        * log loss if y_pred is a probability w/intermediate probabilities
        * sensitivity
        * specificity
        * Positive likelihood ratio (LR_plus)
        * Negative Likelihood ratio (LR_minus)
        * Diagnostic Odds Ratio (DOR)
        * PPV
        * NPV
        * positive rate

    Args:
        y_true: Sample outcomes. Values can be True/False or 1/0 for case/control,
            resp., for each sample
        y_pred: Sample probabilities or class predictions  corresponding to
            y_true.  If class predictions, must have values that match the set of
            values in y_true.
        prob_thres: If y_pred are probabilities, threshold needs to be given to
            convert to classification. Value is None when y_pred are class predictions.
            Default None.
        n_boot_iters: number of bootstrap iterations to estimate 95% CI.
            Default is 10_000.
        alpha_bootstrap: If given, set the alpha for the confidence interval to
            calculate from the bootstrap. Default is 0.05.

    Returns:
        Dictionary keyed by metrics, including mean, medians, and 95% CI estimates by
        bootstrap

    """
    assert (isinstance(n_boot_iters, int)) and (
        n_boot_iters > 0
    ), "Number of bootstraps must be a positive integer"
    alpha_bootstrap_two_tail = alpha_bootstrap / 2

    # Get sample metrics
    actual_metrics = get_binary_classifier_performance_stats(y_true, y_pred, prob_thres)
    actual_metrics = {f"{key}_actual": value for key, value in actual_metrics.items()}

    pred_is_prob = not set(y_pred).issubset({True, False})

    # Bootstrapped sample metrics
    boot_metrics_rows = {}
    for seed_idx in range(n_boot_iters):
        data = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        boot_sample = data.sample(
            n=len(data), replace=True, random_state=seed_idx
        ).reset_index(drop=True)
        seed_metrics = get_binary_classifier_performance_stats(
            boot_sample["y_true"], boot_sample["y_pred"], prob_thres
        )
        boot_metrics_rows[seed_idx] = {
            f"{key}_boot": value for key, value in seed_metrics.items()
        }
    boot_metrics = pd.DataFrame.from_dict(boot_metrics_rows, orient="index")

    # Aggregate metrics across bootstraps
    boot_metrics_agg = {}
    agg_terms = [
        "n_cases_boot",
        "n_controls_boot",
        "prevalence_boot",
        "log_loss_boot",
        "sensitivity_boot",
        "specificity_boot",
        "lr_plus_boot",
        "lr_minus_boot",
        "DOR_boot",
        "PPV_boot",
        "NPV_boot",
        "positive_rate_boot",
    ]
    if pred_is_prob:
        agg_terms += ["AUC_boot", "fpr_interp_boot", "tpr_interp_boot"]
    with np.errstate(divide="ignore", invalid="ignore"):
        for term in agg_terms:
            # Handle column of lists
            bmt = boot_metrics[term]
            if term in ["fpr_interp_boot", "tpr_interp_boot"]:
                bmt_exploded = bmt.explode().astype(float)
                boot_metrics_agg[term + "_median"] = bmt_exploded.median(axis=0)
                boot_metrics_agg[term + "_mean"] = bmt_exploded.mean(axis=0)
                boot_metrics_agg[term + "_se"] = bmt_exploded.std(axis=0)
                boot_metrics_agg[term + "_CI_low"] = bmt_exploded.quantile(
                    q=alpha_bootstrap_two_tail
                )
                boot_metrics_agg[term + "_CI_high"] = bmt_exploded.quantile(
                    q=(1 - alpha_bootstrap_two_tail)
                )
            else:
                if bmt.isna().all():
                    boot_metrics_agg[term + "_median"] = np.nan
                    boot_metrics_agg[term + "_mean"] = np.nan
                    boot_metrics_agg[term + "_se"] = np.nan
                    boot_metrics_agg[term + "_CI_low"] = np.nan
                    boot_metrics_agg[term + "_CI_high"] = np.nan
                else:
                    boot_metrics_agg[term + "_median"] = bmt.median()
                    boot_metrics_agg[term + "_mean"] = bmt.mean()
                    boot_metrics_agg[term + "_se"] = bmt.std()
                    boot_metrics_agg[term + "_CI_low"] = bmt.quantile(
                        q=alpha_bootstrap_two_tail
                    )
                    boot_metrics_agg[term + "_CI_high"] = bmt.quantile(
                        q=(1 - alpha_bootstrap_two_tail)
                    )

    # Aggregate output
    metrics = {**actual_metrics, **boot_metrics_agg}
    return metrics


def get_binary_classifier_performance_across_subsets_with_ci(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    subgroup_filters: pd.DataFrame | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    For each boolean filter column in subgroup_filters, calculate the key stats &
    performance metrics performance in that subgroup with
    mirpy.stats.get_binary_classifier_performance_stats_with_ci().

    Args:
        y_true: Sample outcomes. Values can be True/False or 1/0 for case/control,
            resp., for each sample
        y_pred: Sample probabilities or class predictions  corresponding to
            y_true.  If class predictions, must have values that match the set of
            values in y_true.
        subgroup_filters: A pd.DataFrame where each column defines a subset. The
            column contains a set of True/False or 1/0 entries to filter "y_true" and
            "y_pred_{model_name}" in model_outputs before calculating metrics.
            If None, no samples are applied and subset is named 'all'. Default is None.
        kwargs: kwargs pass through for
            mirpy.stats.get_binary_classifier_performance_stats_with_ci()

    Returns:
        DataFrame of metrics where rows are the subset type and columns are the
        metrics

    """
    if subgroup_filters is None:
        subgroup_filters = pd.DataFrame({"all": [True] * len(y_true)})

    assert (
        len(y_true) == len(y_pred) == len(subgroup_filters)
    ), "Mismatch across number of samples in y_true, y_pred, subgroup_filters"

    assert set(subgroup_filters.values.ravel("K")) <= {
        True,
        False,
    }, "Only True or False values are valid in each subgroup_filters"
    assert (
        subgroup_filters.sum(axis=0) > 1
    ).all(), "Each filter must have at least two samples to evaluate"
    assert not (subgroup_filters.isna().any().any()), "Subset filters cannot contain NA"

    metrics = {}
    for subset_name, subset_filter in subgroup_filters.items():
        metrics[subset_name] = get_binary_classifier_performance_stats_with_ci(
            y_true=y_true[subset_filter],
            y_pred=y_pred[subset_filter],
            **kwargs,
        )

    return (
        pd.DataFrame.from_dict(metrics, orient="index")
        .reset_index()
        .rename(columns={"index": "subset"})
    )


def plot_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    legend_annotation: str = "",
    tpr_error_band: dict | None = None,
    add_diagonal: bool = True,
    color="#295799",
    linestyle="-",
    sensitivity_specificity_tuples: list[tuple] | None = None,
    mark_sensitivity_thres: float | None = None,
    mark_specificity_thres: float | None = None,
    fig_size=(5.25, 5),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Generates ROC with 95% CIs estimated by bootstrap. Confidence interval around ROC
    can be added.

    Args:
        fpr: Increasing false positive rates output, such as those generated by
            sklearn.metrics.roc_curve()
        tpr: Increasing true positive rates corresponding to fpr, such as those
            generated by sklearn.metrics.roc_curve()
        legend_annotation: This string is added as a label for the ROC curve
            corresponding to the tpr vs fpr trendline displayed in the legend. This is
            handy for adding statistics into the legend, such as number of cases and
            controls, AUC, sensitvity, and specificity metrics. An empty string will
            lead to no labels added to the legend for this ROC curve. Default is an
            empty string.
        tpr_error_band: If given, add the 95% CI for the ROC curve. Keys must be
            {"tpr_95_low", "tpr_95_high"} and values are increasing tpr positive rates
            at 95% CI low & high thresholds by bootstrap, resp. Default is None.
        add_diagonal: If True, add diagonal line representing a random classifier
        color: Color of ROC curve and sensitivity/specificity markers, if requested.
            Default is "Mirvie Blue".
        linestyle: Linestyle for ROC curve. Default is solid.
        sensitivity_specificity_tuples: Pairs of sensitivity, specificity tuples to mark
            on the ROC curve. Default is None.
        mark_sensitivity_thres: If given, position of a horizontal line to demarcate
            a sensitivity threshold. Default is None.
        mark_specificity_thres: If given, position of a vertical line to demarcate
            a specificity threshold. Note that the ROC line is generated by plotting tpr
            versus fpr on the x-axis. Specificity thresholds specified would fall on the
            x-axis at the position 1-fpr. Default is None.
        fig_size: Size of figure. Default is (6, 6).
        ax: var for current Matplotlib Axes object. Default is None.

    Returns:
        Current Matplotlib Axes object
    """
    assert len(fpr) == len(tpr), "fpr and tpr must be the same length"
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=fig_size)
    if add_diagonal:
        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            lw=2,
            color="gray",
            label="Random",
            alpha=0.8,
        )
    if mark_sensitivity_thres:
        assert (
            0 <= mark_sensitivity_thres <= 1
        ), "mark_sensitivity_thres must be between 0 and 1"
        plt.axhline(y=mark_sensitivity_thres, color="gray", linestyle=":")
    if mark_specificity_thres:
        assert (
            0 <= mark_specificity_thres <= 1
        ), "mark_specificity_thres must be between 0 and 1"
        plt.axvline(x=1 - mark_specificity_thres, color="gray", linestyle=":")

    plt.plot(
        fpr,
        tpr,
        color=color,
        linestyle=linestyle,
        label=legend_annotation,
        lw=2,
        alpha=0.8,
    )

    if tpr_error_band:
        fill_keys = {"tpr_95_low", "tpr_95_high"}
        assert (
            set(tpr_error_band.keys()) == fill_keys
        ), f"Missing keys in tpr_error_band: {fill_keys- set(tpr_error_band.keys())}"
        for key in fill_keys:
            assert len(tpr_error_band[key]) == len(
                fpr
            ), f"Mismatch length for tpr_error_band {key} to fpr"
        plt.fill_between(
            fpr,
            tpr_error_band["tpr_95_low"],
            tpr_error_band["tpr_95_high"],
            color=color,
            alpha=0.2,
            label="95% CI",
        )

    if sensitivity_specificity_tuples:
        for sens, spec in sensitivity_specificity_tuples:
            plt.plot(1 - spec, sens, color=color, marker="x", alpha=0.8)

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(bbox_to_anchor=[1, 1])
    return ax


def _generate_data_for_roc_by_model_subset(
    model_outputs: pd.DataFrame,
    prob_thresholds: dict,
    subgroup_filters: pd.DataFrame | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Filters the model outputs by each subgroup_filter defined. For each subset,
    binary classifier performance metrics are calculated, with 95% CI metrics
    (as well as other aggregate statistics) estimated by bootstrap.

    Args:
        model_outputs: Dataframe where rows are samples, and columns contain the
            model outcome "y_true" and probability predictions for each model as
            "y_pred_{model_name}"
        prob_thresholds: Keyed by each model_name matching those in model_outputs,
            values are model thresholds for converting probability predictions in
            model_outputs into a classification.
        subgroup_filters: A pd.DataFrame where each column defines a subset. The
            column contains a set of True/False or 1/0 entries to filter "y_true" and
            "y_pred_{model_name}" in model_outputs before calculating metrics.
            If None, no samples are applied and subset is named 'all'. Default is None.
        kwargs: Keyword arguments pass through to
            mirpy.stats.get_binary_classifier_performance_stats()

    Returns:
        Dataframe where rows are each model for each subgroup_filters, where columns
        include model_name, subset, and other columns for performance
        metrics such as auc, tpr, fpr, etc.
    """
    model_names = list(prob_thresholds.keys())
    needed_cols = set(["y_true"] + [f"y_prob_{name}" for name in model_names])
    assert needed_cols <= set(
        model_outputs.columns
    ), "Missing columns in model_outputs: " + str(
        needed_cols.difference(model_outputs.columns)
    )

    output: list = [np.nan] * len(model_names)
    for name_idx, name in enumerate(model_names):
        result = get_binary_classifier_performance_across_subsets_with_ci(
            y_true=model_outputs["y_true"],
            y_pred=model_outputs[f"y_prob_{name}"],
            subgroup_filters=subgroup_filters,
            prob_thres=prob_thresholds[name],
            **kwargs,
        )
        result["model_name"] = name
        output[name_idx] = result

    return pd.concat(output).reset_index(drop=True)


def _round_dict(numerics: dict, digits: int) -> dict:
    """Prints each numeric in dict to the number of digits after decimal"""
    assert isinstance(numerics, dict), "numerics must be a dictionary"
    assert isinstance(digits, int), "digits must be an integer"
    return {key: format(numerics[key], f".{digits}f") for key in numerics}


def plot_roc_by_model_subset(
    model_outputs: pd.DataFrame,
    prob_thresholds: dict,
    subgroup_filters: pd.DataFrame | None = None,
    n_boot_iters: int | None = None,
    alpha_bootstrap: float = 0.05,
    n_multi_test: int = 1,
    color_dict: dict | None = None,
    linestyles: dict | None = None,
    add_metrics_to_legend: bool = True,
    mark_sensitivity_thres: float | None = 0.7,
    mark_specificity_thres: float | None = 0.7,
    sens_spec_thresholds: Sequence | None = (
        (0.7, 0.7),
        (0.7, 0.69),
        (0.75, 0.7),
        (0.80, 0.7),
    ),
    title: str | None = None,
    fig_size: list | tuple = (5.25, 5),
) -> plt.Axes:
    """
    Plots ROC for one or more model types. Can be used for a single train/test or
    applied to a cross validation with many splits. Annotations metrics annotated in
    legend. Summary statistics printed to log.

    Args:
        model_outputs:
            Dataframe where rows are samples, and columns contain the
            model outcome "y_true" and probability predictions for each model as
            "y_pred_{model_name}"
        prob_thresholds: Keyed by each model_name matching those in model_outputs,
            values are model thresholds for converting probability predictions in
            model_outputs into a classification.
        subgroup_filters: A pd.DataFrame where each column defines a subset. The
            column contains a set of True/False or 1/0 entries to filter "y_true" and
            "y_pred_{model_name}" in model_outputs before calculating metrics.
            If None, no samples are applied and subset is named 'all'. Default is None.
        n_boot_iters: Number of bootstrap iterations to estimate 95% CI for key metrics
            calculated in mirpy.stats.get_binary_classifier_performance_stats(). If
            None, default setting in
            mirpy.stats.get_binary_classifier_performance_stats() is applied. Default
            is None.
        alpha_bootstrap: If given, set the alpha for the confidence interval to
            calculate from the bootstrap for key metrics
            calculated in mirpy.stats.get_binary_classifier_performance_stats().
            Default is 0.05.
        n_multi_test: If given, perform a multiple test correction to adjust
            alpha_bootstrap based on a Bonferoni correction
            (alpha_boostrap / n_multi_test) for key metrics
            calculated in mirpy.stats.get_binary_classifier_performance_stats().
            Default sets number of tests to 1.
        color_dict: Dictionary keyed by the "subset" with values for
            the color to use for that subset's ROC curve in plot. Default is None, which
            applies the color colors from the 'cividis' color map.
        linestyles: Dictionary keyed by the "model_name" with values for
            linestyles to use for that subset's ROC curve in plot. If None, solid lines
            are used for all models. Default is None
        add_metrics_to_legend: If True, add metrics for each model as a label for its
            ROC in the legend. If False, only label with model name. Default is True.
        mark_sensitivity_thres: If given, position of the horizontal line to demarcate
            a sensitivity threshold. Default is 0.7.
        mark_specificity_thres: If given, position of the vertical line to demarcate
            a specificity threshold. Default is 0.7.
        sens_spec_thresholds: List of sensitivity, specificity thresholds to evaluate
            each model subset performance against. This prints to the console  whether
            the sensitivity and specificity values are over these thresholds to
            console, without impacting the figure. If None, no readout is provided.
            Default is None.
        title: If given, add this title. If None, default title is
            "{model} performance}" with each subset added in a list behind a colon.
            Default is None.
        fig_size: Size of figure. Default is (5.25, 5).

    Returns:
        Current Matplotlib Axes object

    Examples:
        >>> import mirpy.data as md
        >>> from mirpy.viz import plot_roc_by_model_subset
        >>>
        >>> penguins = md.get_palmer_penguins_data()
        >>> penguins_bin = penguins[penguins["species"].isin(["Chinstrap", "Gentoo"])
        >>>                         ].reset_index(drop=True)
        >>> penguins_bin["is_gentoo"] = penguins_bin["species"] == "Gentoo"
        >>> penguins_bin["prob"] = (
        >>>     1 -
        >>>     penguins_bin["bill_length_mm"] / penguins_bin["bill_length_mm"].max()
        >>> )
        >>> penguins_bin["is_male"] = penguins_bin["sex"] == "male"
        >>> penguins_bin["is_female"] = ~penguins_bin["is_male"]
        >>>
        >>> plot_roc_by_model_subset(
        >>>     model_outputs=penguins_bin[["is_gentoo", "prob"]].rename(
        >>>         columns={'is_gentoo': 'y_true', 'prob': 'y_prob_full'}),
        >>>     prob_thresholds={"full": 0.2},
        >>>     subgroup_filters=penguins_bin[["is_male","is_female"]],
        >>>     n_boot_iters=100,
        >>>     color_dict ={'is_male': '#377EB8',
        >>>                 'is_female': '#ff7f00'},
        >>>     linestyles ={'full': '-'},
        >>>     mark_sensitivity_thres=None,
        >>>     mark_specificity_thres=None,
        >>>     sens_spec_thresholds = None
        >>> )
    """
    assert isinstance(alpha_bootstrap, float) and (
        0 < alpha_bootstrap < 1
    ), "alpha_bootstrap must be a float between 0 and 1"
    if n_multi_test is None:
        n_multi_test = 1
    assert isinstance(n_multi_test, int) and (
        n_multi_test >= 1
    ), "n_multi_test must be an integer value >= 1"
    alpha_bootstrap_corr = alpha_bootstrap / n_multi_test

    bs_kwargs = {"alpha_bootstrap": alpha_bootstrap_corr}
    if n_boot_iters:
        bs_kwargs = {**bs_kwargs, "n_boot_iters": n_boot_iters}

    roc_data = _generate_data_for_roc_by_model_subset(
        model_outputs=model_outputs,
        prob_thresholds=prob_thresholds,
        subgroup_filters=subgroup_filters,
        **bs_kwargs,
    )

    if color_dict:
        assert set(roc_data["subset"]) <= set(color_dict.keys()), (
            "color_dict is missing a color assignment for these subsets: "
            f"{set(set(roc_data['subset']) - color_dict.keys())}"
        )
    else:
        names = roc_data["subset"].unique()
        cmap = plt.get_cmap("cividis", len(names))
        palette = [rgb2hex(cmap(i)) for i in range(cmap.N)]
        color_dict = {name: palette[idx_name] for idx_name, name in enumerate(names)}

    if linestyles:
        assert set(roc_data["model_name"]) <= set(linestyles.keys()), (
            "linestyles is missing a linestyle assignment for these model names: "
            f"{set(roc_data['model_name'])-set(linestyles.keys())}"
        )
    else:
        linestyles = {name: "-" for name in roc_data["model_name"].unique()}

    # Plot curves on ROC for each model
    for model_idx, row in roc_data.iterrows():
        model_name = row.model_name
        subset_name = row.subset
        # ROC settings
        if model_idx == 0:
            ax = None
            add_diagonal = True
        else:
            add_diagonal = False

        info_auc = (
            "\nAUC {AUC_actual} [{ci_interval}% CI: "
            + "{AUC_boot_CI_low} - {AUC_boot_CI_high}]"
        )
        ci_interval = (1 - alpha_bootstrap_corr) * 100
        numeric_auc = {
            "AUC_actual": row.AUC_actual,
            "ci_interval": ci_interval,
            "AUC_boot_CI_low": row.AUC_boot_CI_low,
            "AUC_boot_CI_high": row.AUC_boot_CI_high,
        }
        info = (
            "\nPrevalence {prevalence_actual}%"
            + "\nSensitivity {sensitivity_actual} "
            + "[{ci_interval}% CI: "
            + "{sensitivity_boot_CI_low} - {sensitivity_boot_CI_high}]"
            + "\nSpecificity {specificity_actual} "
            + "[{ci_interval}% CI: "
            + "{specificity_boot_CI_low} - {specificity_boot_CI_high}]"
        )
        numerics = {
            "ci_interval": ci_interval,
            "prevalence_actual": row.prevalence_actual,
            "sensitivity_actual": row.sensitivity_actual,
            "sensitivity_boot_CI_low": row.sensitivity_boot_CI_low,
            "sensitivity_boot_CI_high": row.sensitivity_boot_CI_high,
            "specificity_actual": row.specificity_actual,
            "specificity_boot_CI_low": row.specificity_boot_CI_low,
            "specificity_boot_CI_high": row.specificity_boot_CI_high,
        }

        stat_label = {}
        for precision in [2, 6]:
            rounded_aucs = _round_dict(numeric_auc, precision)
            rounded_aucs["ci_interval"] = (
                rounded_aucs["ci_interval"].rstrip("0").rstrip(".")
            )
            rounded_numerics = _round_dict(numerics, precision)
            rounded_numerics["ci_interval"] = (
                rounded_numerics["ci_interval"].rstrip("0").rstrip(".")
            )
            stat_label[precision] = (
                f"\n{row.model_name.title()}, {row.subset.title()}"
                + info_auc.format(**rounded_aucs)
                + f"\nNo. cases {int(row.n_cases_actual):d}"
                + f"\nNo. controls {int(row.n_controls_actual):d}"
                + info.format(**rounded_numerics)
            )

        # Log metrics at high precision (low precision wil be added to legend)
        print(stat_label[6])
        if add_metrics_to_legend:
            legend_anno = stat_label[2]
        else:
            legend_anno = f"{row.model_name.title()}, {row.subset.title()}"
        if sens_spec_thresholds:
            for sens, spec in sens_spec_thresholds:
                pass_sens_spec = (row.sensitivity_actual > sens) & (
                    row.specificity_actual > spec
                )
                print(f"{sens}/{spec}  pass: {pass_sens_spec}")

        ax = plot_roc(
            fpr=row.fpr_actual,
            tpr=row.tpr_actual,
            legend_annotation=legend_anno,
            tpr_error_band=None,
            add_diagonal=add_diagonal,
            color=color_dict[subset_name],
            linestyle=linestyles[model_name],
            sensitivity_specificity_tuples=[
                (row.sensitivity_actual, row.specificity_actual)
            ],
            mark_sensitivity_thres=mark_sensitivity_thres,
            mark_specificity_thres=mark_specificity_thres,
            fig_size=fig_size,
            ax=ax,
        )

    if title is None:
        title = ""
        for idx_model, model_name in enumerate(set(roc_data["model_name"])):
            if idx_model == 0:
                delimiter = ""
            else:
                delimiter = ", "
            title = title + f"{delimiter}{model_name.title()}"
        title = title + "performance"

        for idx_subset, subset in enumerate(set(roc_data["subset"])):  # All is a subset
            if idx_subset == 0:
                delimiter = ":"
            else:
                delimiter = ","
            title = title + f"{delimiter} {subset.title()}"
    plt.title(title)

    return ax  # type: ignore[return-value]
