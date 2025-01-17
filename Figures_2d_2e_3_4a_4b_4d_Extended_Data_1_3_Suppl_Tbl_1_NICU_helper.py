from collections import OrderedDict
from itertools import chain, tee
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import chi2, linregress, mannwhitneyu
from sklearn.linear_model import HuberRegressor
from statsmodels.stats.multitest import multipletests


def _add_pvalue_bar_to_plot(
    pval: float,
    x_groups: list[tuple[str, str]],
    ax: plt.Axes,
    max_y: float,
    height: float,
    height_level: int,
    color: str = "dimgray",
    pct_offset: float = 0.05,
):
    """
    Add p-value annotation (bar & p-value) to a boxplot between two groups mapped to the
    X axis, specified by each tuple pair in `x_groups`.

    Args:
        pval: p-value to use in bar annotation
        x_groups: list of 2-tuples group names in x-axis to add p-value bar across
            e.g. ``[("A", "B"), ("B", "C"), ("A", "C")]``
        ax: axes class for plot to annotate
        max_y: max y-value of data in plot
        height: height of bars in plot
        height_level: level at which to add the bar (used to add multiple p-value
            bars to a pot). Default is 1.
        color: color of p-value bar annotation. Default is "dimgray".
        pct_offset: number between -1 and 1; percent to shrink bar annotation edges
            at x-axis groups (0 is centered). Default is 0.05 (5%).
    """
    assert (pval >= 0) & (
        pval <= 1
    ), f"pval must be in the interval [0,1], given {pval}"
    assert len(x_groups) == 2, "Only 2 group names on the x-axis can be annotated"
    # Get position on x axis
    x_pos = [np.nan] * 2
    for group_idx, group_name in enumerate(x_groups):
        x_match = [x for x in ax.get_xticklabels() if x.get_text() == str(group_name)]
        assert len(x_match) == 1, f"Single match for {group_name} not found"
        x_pos[group_idx] = x_match[0].get_position()[0]
    x_pos.sort()
    # Add offsets to prevent adjacent annotations from overlapping
    x_pos[0] = (1 + pct_offset) * x_pos[0]
    x_pos[1] = (1 - pct_offset) * x_pos[1]

    # Set bar params
    bar_y = max_y + height * height_level + height * (height_level - 1)

    # Convert p-value to text annotation
    if pval > 0.001:
        text = f"{pval:.3f}".rstrip("0")
    else:
        text = f"p={pval:.1e}"

    # Add bar with annotation
    ax.plot(
        [x_pos[0], x_pos[0], x_pos[1], x_pos[1]],
        [bar_y, bar_y + height, bar_y + height, bar_y],
        lw=1.5,
        c=color,
    )
    ax.text(
        (x_pos[0] + x_pos[1]) * 0.5,
        (bar_y + height) + 0.2 * height,
        str(text),
        ha="center",
        va="bottom",
        color=color,
    )

    # Adjust ylims to fit bar
    y_lims = ax.get_ylim()
    ax.set_ylim(bottom=y_lims[0], top=max(y_lims[1], bar_y + 3.5 * height))


def plot_boxplots_with_pvals(
    plot_df: pd.DataFrame,
    x: str,
    y: str,
    order: list,
    contrasts: list[tuple[str, str]],
    fig_size: list | tuple = (5, 3.75),
    title: str | None = None,
    p_sig_thresh: float = 0.05,
    mw_alt: str = "two-sided",
    multiple_test_corr_method: str = "bonferroni",
    palette: str | sns.palettes._ColorPalette | None = None,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    Plot boxplot with p-value bars calculated for each tuple in contrasts. P-values
    are adjusted by Mann Whitney where the alternative is defined by mw_alt and
    multiple test correction multiple_test_corr_method is applied.

    Args:
        plot_df: Dataframe that contains x and y to plot
        x: column name of plot_df to plot on x-axis (should be a string or category
            column type)
        y: column name of plot_df to plot on y-axis (should be a numeric column type)
        order: order of groups on x-axis, from left to right
        contrasts: list of 2-tuples group names in x-axis to add p-value bar across
                e.g. ``[("A", "B"), ("B", "C"), ("A", "C")]``.
                This function annotates pairs sequentially through the list. Annotations
                will appear at the lowest available level where the space is free.
                Recommended usage is to submit contrasts for adjacent/neighboring groups
                first (shorter bars) followed by pairs with increasing distance between
                them (longer bars).
        fig_size: width, height  of figure, in inches. Default is (5, 3.75).
        title: if given, add title. Default is None.
        p_sig_thresh: significance level for multiple testing pass/fail. Default is
            0.05.
        mw_alt: alternative hypothesis for scipy.stats.mannwhitneyu. Default is
            two-sided.
        multiple_test_corr_method: mutiple testing correction method for
            statsmodels.stats.multitest import multipletests. Default is bonferroni.
        palette: if None, use standard palette for boxplot. Default is
            reversed(sns.color_palette("viridis").

    Returns:
        figure handle, figure Axes, pd.DataFrame containing p-values from
        adjusted Mann-Whitney p-values
    """
    assert isinstance(contrasts, list), "Contrast must be a list of tuples"
    assert all(
        isinstance(x, tuple) for x in contrasts
    ), "Contrast must be a list of tuples"
    all_contrasts = list(set(chain(*contrasts)))
    assert set(all_contrasts) <= set(
        plot_df[x]
    ), "Some elements of contrasts are not in plot_df[x]"
    assert set(order) <= set(
        plot_df[x]
    ), "Order must be a subset or equal to the unique values of x"
    assert not (plot_df[x].isna().any()), "plot_df[x] cannot contain missing values"
    if palette is None:
        palette = reversed(sns.color_palette("viridis"))
    elif isinstance(palette, str):
        palette = sns.color_palette(palette, as_cmap=False)

    # Take just needed rows and columns, make a deep copy to prevent changing original
    df = plot_df.loc[plot_df[x].isin(order), [x, y]].reset_index(drop=True).copy()

    # sort df by x column to control color assigment, using the order variable
    df = df.sort_values(by=x, key=lambda column: column.map(lambda e: order.index(e)))

    # Make boxplot
    fig, ax = plt.subplots(figsize=fig_size)
    ax.grid(False)
    palette, palette_points = tee(palette)  # copy iterator to prevent consumption
    sns.stripplot(
        data=df,
        x=x,
        y=y,
        order=order,
        hue=x,
        dodge=False,
        palette=palette_points,
        legend=False,
    )
    sns.boxplot(
        data=df,
        x=x,
        y=y,
        order=order,
        hue=x,
        boxprops={"alpha": 0.3},
        whiskerprops={"linewidth": 0.7},
        palette=palette,
        legend=False,
    )
    # Add plot labels
    if title is None:
        title = ""
    else:
        assert isinstance(title, str), "Title must be a string"
    ax.set_title(title)

    # Add p-value annotations
    mw_rows = {}
    for i, contrast in enumerate(contrasts):
        mw_df = df[df[x].isin(contrast)].reset_index(drop=True).copy()
        mw_df["mw_label"] = mw_df[x] == contrast[0]
        _, pval = mannwhitneyu(
            mw_df.loc[mw_df["mw_label"] == False, y].reset_index(drop=True),
            mw_df.loc[mw_df["mw_label"] == True, y].reset_index(drop=True),
            alternative=mw_alt,
            use_continuity=False,
        )
        mw_rows[i] = {
            "contrast_a": contrast[0],
            "contrast_b": contrast[1],
            "pvalue": pval,
        }
    mw = pd.DataFrame.from_dict(mw_rows, orient="index")
    mw["reject_null"], mw["pvalue_adj"], _, _ = multipletests(
        mw.pvalue, alpha=p_sig_thresh, method=multiple_test_corr_method
    )
    ylims = ax.get_ylim()
    height = (ylims[1] - ylims[0]) * 0.1

    # Plot pval bars in lowest available space
    rendered_positions: OrderedDict[int, set] = OrderedDict({1: set()})
    for _, row in mw.iterrows():
        pval_adj = row.pvalue_adj
        x_group_1 = row.contrast_a
        x_group_2 = row.contrast_b
        x_group_1_position = order.index(x_group_1)
        x_group_2_position = order.index(x_group_2)
        # The actual positions that need to be tracked are traversals between two
        # groups, e.g. the span between 1 and 3 (not inclusive). For convenience, and to
        # remain in integer representations, these spans are represented in the tracking
        # sets by the integer floor of the traversed numbers, e.g. span between 1-3
        # is represented by the set (1, 2).
        utilized_x_spaces = set(
            range(
                min(x_group_1_position, x_group_2_position),
                max(x_group_1_position, x_group_2_position),
            )
        )
        annotation_not_placed = True
        candidate_height = 1
        while annotation_not_placed:
            if candidate_height not in rendered_positions:
                rendered_positions[candidate_height] = set()
            if rendered_positions[candidate_height].isdisjoint(utilized_x_spaces):
                rendered_positions[candidate_height] = rendered_positions[
                    candidate_height
                ].union(utilized_x_spaces)
                annotation_not_placed = False
                break
            candidate_height += 1
        _add_pvalue_bar_to_plot(
            pval=pval_adj,
            x_groups=[x_group_1, x_group_2],
            ax=ax,
            max_y=df[y].max(),
            height=height,
            height_level=candidate_height,
        )

    # Add sample size annotations to x-axis
    counts = df[x].value_counts(sort=False)
    labels = [
        item.get_text() + f"\n(n={counts[order[idx]]})"
        for idx, item in enumerate(ax.get_xticklabels())
    ]
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90)
    return fig, ax, mw


def scatter_plot_with_fits(
    df: pd.DataFrame,
    x_label: str,
    x_vanity_label: str,
    y_label: str,
    y_vanity_label: str,
    prediction_label: str,
    prediction_vanity_label: str,
):
    """
    Plot scatter plot with linear lease-squares regression fits. Fit annotations added
    to legend.

    Args:
        df: input data for plot
        x_label: column in df to plot on x-axis
        x_vanity_label: label printed on the x-axis
        y_label: column in df to plot on y-axis
        y_vanity_label: label printed on the y-axis
        prediction_label: case/control label where values are True/False, resp. in
            which to split scatter plot. each group is plot as a different hues
        prediction_vanity_label: label to print in lieu of prediction_label

    """
    # Setup figure
    sns.set_style(
        style="white",
        rc={"figure.figsize": [4, 2.5], "font.size": 16, "lines.markersize": 7},
    )

    # Colors from tab10
    cm_colors = [
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
        (1.0, 0.4980392156862745, 0.054901960784313725),
    ]

    plot_df = (
        df.dropna(subset=[x_label, y_label, prediction_label])
        .reset_index(drop=True)
        .copy()
    )
    assert set(plot_df[prediction_label]) <= {
        False,
        True,
    }, (
        f"Values for {prediction_label} must be True and/or False, "
        + f"given {set(plot_df[prediction_label])}"
    )

    # Get Annotations for control/case
    count_df = plot_df[prediction_label].value_counts()
    rsq_map = {}
    slope_map = {}
    pval_map = {}
    for label_val in [False, True]:
        if label_val not in count_df.index:
            count_df = pd.concat([count_df, pd.Series({label_val: np.nan})])
            rsq_map[label_val] = np.nan
            pval_map[label_val] = np.nan
            slope_map[label_val] = np.nan
            continue
        eval_df = plot_df.loc[plot_df[prediction_label] == label_val].reset_index(
            drop=True
        )
        x = eval_df[x_label]
        y = eval_df[y_label]
        m, b, r, p, se = linregress(x, y)
        rsq_map[label_val] = r**2
        pval_map[label_val] = p
        slope_map[label_val] = m

    print(f"{x_vanity_label}, {y_vanity_label}, r2: {rsq_map}\n")

    # Reformat exponential notation to scientific notation with powers of 10
    p_string_pos = f"{pval_map[True]:.1e}"
    p_string_pos_comp = p_string_pos.split("e-")
    p_string_pos_anno = "{0}x$10^{{{1}}}$".format(
        p_string_pos_comp[0], -int(p_string_pos_comp[1])
    )

    name_map = {
        False: f"Not {prediction_vanity_label}\nn={count_df[False]}, "
        + f"$r^2$={rsq_map[False]:.2f},\n"
        + f"m={slope_map[False]:.0e} (p={pval_map[False]:.2f})",
        True: f"{prediction_vanity_label}\nn={count_df[True]}, "
        + f"$r^2$={rsq_map[True]:.2f},\n"
        + f"m={slope_map[True]:.2f} (p={p_string_pos_anno})",
    }
    plot_df["hue_label"] = plot_df[prediction_label].map(name_map)

    # Add regression plot & contours
    df_cases = plot_df[plot_df[prediction_label] == True].copy()
    x_subset_T = pd.Series(df_cases[x_label], name=f"{x_label}_subset_T")
    y_subset_T = pd.Series(df_cases[y_label], name=f"{y_label}_subset_T")

    df_controls = plot_df[plot_df[prediction_label] == False].copy()
    x_subset_F = pd.Series(df_controls[x_label], name=f"{x_label}_subset_F")
    y_subset_F = pd.Series(df_controls[y_label], name=f"{y_label}_subset_F")

    plot_df = pd.concat(
        [plot_df, x_subset_T, y_subset_T, x_subset_F, y_subset_F], axis=1
    )

    g = sns.FacetGrid(data=plot_df, height=4.5, aspect=2)

    for x, y, color, label in [
        (f"{x_label}_subset_T", f"{y_label}_subset_T", cm_colors[1], name_map[True]),
        (f"{x_label}_subset_F", f"{y_label}_subset_F", cm_colors[0], name_map[False]),
    ]:
        if plot_df[y].isna().all():
            print(f"No samples apply for {label}, skipping")
            continue
        g.map(
            sns.regplot,
            x,
            y,
            color=color,
            scatter=False,
            label=label,
        )

        g.map_dataframe(
            sns.kdeplot,
            x=x,
            y=y,
            color=color,
            fill=True,
            alpha=0.3,
            levels=5,
        )

    # Only add points for cases if less than 200 samples
    n_dots = len(plot_df[f"{x_label}_subset_T"].dropna())
    print(f"\nNumber of scatter points to add (skip if >= 200): {n_dots}\n")
    if (n_dots > 0) & (n_dots < 200):
        g.map_dataframe(
            sns.scatterplot,
            x=f"{x_label}_subset_T",
            y=f"{y_label}_subset_T",
            color=cm_colors[1],
            alpha=1,
            marker=".",
            s=27,
        )
    g.add_legend()
    plt.xlabel(x_vanity_label)
    plt.ylabel(y_vanity_label)

    plt.show()


def get_genomic_inflation_factor(pvals: pd.Series | np.ndarray) -> float:
    """
    Calculate genomic inflation factor based on chi-sq 1 df. The genomic
    inflation factor is the median of the chi-square test statistics divided by the
    expected median of the chi-square distribution.
    Args:
        pvals: p-values from which to calculate the genomic inflation factor
    Returns:
        genomic inflation factor
    """
    if isinstance(pvals, pd.Series):
        pvals = pvals.to_numpy()
    return np.median([chi2.ppf(1 - x, 1) for x in pvals]) / chi2.ppf(0.5, 1)


def qqplot(
    data: pd.DataFrame,
    n_cases: int,
    n_controls: int,
    plot_fit: bool | None = None,
    padj_cutoff: float | None = 0.05,
    plt_title: str | None = None,
    fig_width: int | None = 7,
    fig_height: int | None = 5,
    y_lims: tuple | None = None,
    hue_label: str | None = None,
    hue_cmap: str | LinearSegmentedColormap | None = None,
    hue_min: float | None = None,
    hue_max: float | None = None,
    legend_title: str | None = None,
) -> plt.Figure:
    """
    Generates a QQ Plot of -log10 observed p-values (y-axis) vs. expected -log10
    p-values from a uniform distribution (x-axis). The genomic inflation factor (lambda)
    is calculated and reported in the plot title. Lambda is the median of the chi-square
    test statistics divided by the expected median of the chi-square distribution.

    Args:
        data: Dataframe containing the raw ("pvalue") to plot where rows are features
            (names are in column "feature").
            Note: If padj_cutoff is given, multiple test adjusted p-values
            ("pvalue_adj") must be given.
        n_cases: Number of cases used in analysis to calculate p-values.
        n_controls: Number of controls used in analysis to calculate p-values.
        plot_fit: If True, robust fit line to data and add to plot. If False or None,
            no fit is performed. Default is to not perform a fit.
        padj_cutoff: If given, plot a dashed line at observed p_value for which
            "pvalue_adj" in data above line represents those that are significant at
            padj_cutoff. Default is alpha level 0.05.
            Note: This horizontal line is an approximation between observed p-values
            and not an exact observed p-value. The purpose is to give a visual guide
            to display which genes are significant at the provided alpha level.
        plt_title: If given, add title to figure. Default is not title added.
        fig_width: Width of QQ plot, if not given, defaults to 7.
        fig_height: Height of QQ plot, if not given, defaults to 5.
        y_lims: If not None, set y-axis to limits, default is to use the plot range.
        hue_label: If not None, color scatter plot by this label, defaults to no
            color added.
        hue_cmap: If not None, palette to use for hue_label, defaults to "hsv".
        legend_title: Title to add to legend, defaults to no title added.

    Returns:
        fig: matplotlib figure handle

    """
    # Input checks
    req_columns = ["pvalue"]
    if hue_label:
        req_columns.append(hue_label)
    if padj_cutoff:
        req_columns.append("pvalue_adj")
    assert set(req_columns) <= set(
        data.columns
    ), f"Data is missing required columns: {set(req_columns) - set(data.columns)}."

    # subset to features with p-value
    df = data.dropna(subset=["pvalue"]).copy()
    assert len(df) > 0, "QQ plot requires at least one non-NaN entry."
    df.sort_values(by="pvalue", inplace=True, ascending=True)
    df.reset_index(drop=True, inplace=True)

    # Calculate QQ plot
    observ_pval = -np.log10(df.pvalue.to_numpy())
    n = len(observ_pval)

    # Expect uniform(0,1) distribution for no association
    theoretical_pval = -np.log10(np.arange(1 / (n + 1), 1, 1 / (n + 1))[:n])

    # Calculate lambda based on chi-sq 1 df
    lmbda = get_genomic_inflation_factor(df["pvalue"])

    # Calculate cutoff
    last_fdr = df[(df["pvalue_adj"] < padj_cutoff)].last_valid_index()
    if last_fdr is not None:
        last_iloc = df.index.get_loc(last_fdr)
        if last_iloc == df.index[-1]:  # All values pass adjusted pvalue cutoff
            y_cutoff = df["pvalue"].max() * 1.05
        else:
            y_cutoff = (
                df.iloc[last_iloc]["pvalue"] + df.iloc[last_iloc + 1]["pvalue"]
            ) / 2.0
    else:
        y_cutoff = None

    # Create plot
    if hue_cmap:
        cmap = hue_cmap
    else:
        cmap = "hsv"

    if hue_label:
        hue = df[hue_label]
        if not hue_min:
            hue_min = hue.min()
        if not hue_max:
            hue_max = hue.max()

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.scatterplot(
        x=theoretical_pval,
        y=observ_pval,
        hue=hue,
        ax=ax,
        linewidth=0,
        palette=cmap,
        hue_norm=(hue_min, hue_max),
        legend=False,
    )
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(hue_min, hue_max), cmap=cmap),
        ax=ax,
        orientation="vertical",
    ).set_label(label=legend_title, size=14, labelpad=8)
    ax.set_xlabel("expected p-value (-log10)")
    ax.set_ylabel("observed p-value (-log10)")

    if y_lims:
        if (observ_pval.min() < y_lims[0]) | (y_lims[1] < observ_pval.max()):
            print(
                "WARNING! Y-axis limit does not cover full range of observed p-values."
            )
        plt.ylim(y_lims)

    # Set x-axis to start at 0
    xlims = ax.get_xlim()
    xlims = [0, xlims[1]]
    plt.xlim(xlims)

    # Add adjusted pvalue line
    if padj_cutoff and y_cutoff:
        ax.axhline(
            y=-np.log10(y_cutoff),
            color="lightgray",
            linestyle="dashed",
            label=f"$p_{{adj}}<{padj_cutoff}$",
        )

    # Add robust fit
    if plot_fit:
        huber = HuberRegressor(epsilon=1.345, max_iter=100).fit(
            theoretical_pval.reshape(-1, 1), observ_pval
        )
        observ_pval_fit = huber.predict(theoretical_pval.reshape(-1, 1))
        sns.lineplot(
            x=theoretical_pval,
            y=observ_pval_fit,
            label="Robust Fit",
            color="silver",
            ax=ax,
        )

    # Finalize plot
    plt.legend(
        loc="upper left",
        frameon=True,
        title=f"{n_cases:d} cases\n{n_controls:d} controls",
        title_fontproperties={"size": 12},
        prop={"size": 12},
    )
    if plt_title:
        plt.title(f"lambda {lmbda:.1f}\n{plt_title}")

    return fig


def make_custom_seq_colormap(
    color_A: str | Sequence[float],
    color_B: str | Sequence[float],
    split_point: float,
    map_name: str = "custom",
) -> LinearSegmentedColormap:
    """
    Create a custom colormap with two colors that overlap at a specified split point.
    NOTE: Base color is gray.

    Parameters:
        color_A: first color in RGB format or name (e.g., 'red', 'blue')
        color_B: second color in RGB format or name (e.g., 'red', 'blue')
        split_point: float, position where the colors should blend (0.0 to 1.0)
        map_name: Name of custom color map

    Returns:
        colormap: LinearSegmentedColormap object
    """
    if isinstance(color_A, str):
        color_A = mpl.colors.to_rgb(color_A)

    if isinstance(color_B, str):
        color_B = mpl.colors.to_rgb(color_B)

    cdict = {
        "red": [
            (0.0, 0.5, 0.5),
            (split_point, color_B[0], color_B[0]),
            (1.0, color_A[0], color_A[0]),
        ],
        "green": [
            (0, 0.5, 0.5),
            (split_point, color_B[1], color_B[1]),
            (1.0, color_A[1], color_A[1]),
        ],
        "blue": [
            (0, 0.5, 0.5),
            (split_point, color_B[2], color_B[2]),
            (1.0, color_A[2], color_A[2]),
        ],
    }
    return LinearSegmentedColormap(map_name, cdict)


def plot_qq_set(df: pd.DataFrame, eval_list: Sequence[tuple]):
    """
    Generates a set of QQ plots with a shared y-axis and color map range

    Args:
        df: Dataframe containing features & p-values for one or more analyses,
            differentiated by the "analysis" column.
        eval_list: Each element is a tuple of analysis name and the top features to
            print the p-values (sorted in ascending order)
    """
    required_columns = {
        "feature",
        "pvalue",
        "analysis",
        "n_cases",
        "n_control",
        "cohens_d",
        "abs_effect_size",
    }
    assert required_columns <= set(
        df.columns
    ), f"Missing required columns to plot: {required_columns -set(df.columns)}"

    # Set global parameters across all QQ plots
    y_lims = (0, -np.log10(df["pvalue"].min()) * 1.1)
    hue_max = df["abs_effect_size"].max() * 1.1

    # Generate QQ plot for each analysis
    for name, n_anno in eval_list:
        mw_df = (
            df[df["analysis"] == name]
            .sort_values(by="pvalue", ascending=True)
            .reset_index()
        )
        assert (
            len(mw_df["n_cases"].unique()) == 1
        ), f"Mismatched number of cases in analysis for {name}"
        assert (
            len(mw_df["n_control"].unique()) == 1
        ), f"Mismatched number of cases in analysis for {name}"
        n_cases = mw_df["n_cases"].unique()[0]
        n_controls = mw_df["n_control"].unique()[0]
        _ = qqplot(
            data=mw_df,
            n_cases=n_cases,
            n_controls=n_controls,
            plot_fit=True,
            padj_cutoff=0.05,
            plt_title=name,
            fig_width=7,
            fig_height=5,
            y_lims=y_lims,
            hue_label="abs_effect_size",
            hue_cmap=make_custom_seq_colormap(
                color_A="red", color_B="blue", split_point=0.2
            ),
            hue_min=0,
            hue_max=hue_max,
            legend_title="Absolute Effect Size",
        )
        plt.show()

        print(f'{mw_df.loc[:(n_anno-1), ["feature", "pvalue", "cohens_d"]]}\n\n')
