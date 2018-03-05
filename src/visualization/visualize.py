from collections import OrderedDict
from contextlib import contextmanager
from operator import xor
import os

from cycler import cycler

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import FuncFormatter

import numpy as np

import pandas as pd

import seaborn as sns

PALETTE = OrderedDict([
    ('green', "#198E7D"),
    ('orange', "#E84B00"),
    ('purple', "#54333c"),
    ('red', '#B83539'),
    ('lime', '#8DB134'),
])
sns.set_palette(list(PALETTE.values()))


@contextmanager
def styled_fig_ax(size='wide',
                  font_size=10.0,
                  zero_lines=True,
                  y_axis_grid=True,
                  seaborn=False,
                  subplots_rows=None,
                  subplots_columns=None,
                  subplots_kwargs=None,
                  x_formatter=None,
                  y_formatter=None,
                  other_rc_params=None):
    """ Context manager for a styled axis.

        :param size: accepts 'wide' (top/bottom of slide), 'tall' (left side of slide),
                    'tallest' (left side of slide, lots of text)
        :param font_size: the main font size; some sizes are set relative to this; overridden on tallest.
        :param zero_lines: strong lines at x=0 and y=0
        :param other_rc_params: any custom rcParams to include
        :param subplots_rows: the number of rows in the subplot (default=None)
        :param subplots_columns: the number of columns in the subplot (default=None)
        :param subplots_kwargs: dict of kwargs for subplots call (e.g., sharex=True) (default={})
        :param x_formatter: Formatter to use for major ticks on x axis
        :param y_formatter: Formatter to use for major ticks on y axis
    """
    subplots_kwargs = {} if subplots_kwargs is None else subplots_kwargs
    other_rc_params = {} if other_rc_params is None else other_rc_params

    sizes = {
        'quarter': (6.8, 3.7),
        'wide': (9.75, 3.6),
        'tall': (6., 7.5),
        'tallest': (6, 8.5),
        'square': (6., 6.),
        'custom': other_rc_params.get('figure.figsize', None),
    }
    figure_size = sizes[size]

    original_params = plt.rcParams.copy()

    if sizes == 'tallest':
        plt.rcParams['savefig.pad_inches'] = 0.0
        plt.rcParams['savefig.pad_inches'] = 0.0

        if font_size > 6.0:
            font_size = 6.0

    # set globally here since this seems ignored by rc_context.....
    # apologies for side effects.
    plt.rcParams['axes.titlesize'] = 1.25 * font_size
    plt.rcParams['figure.dpi'] = 250
    plt.rcParams['savefig.dpi'] = 250
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.fontsize'] = 0.8 * font_size

    rc_params = {
        'figure.figsize': figure_size,

        # font
        'font.family': 'Verdana',
        'font.size': font_size,
        'axes.labelsize': font_size,
        'xtick.labelsize': 0.8 * font_size,
        'ytick.labelsize': 0.8 * font_size,

        # remove extras
        'xtick.major.size': 0,      # major tick size in points
        'xtick.minor.size': 0,      # minor tick size in points
        'ytick.major.size': 0,      # major tick size in points
        'ytick.minor.size': 0,      # minor tick size in points

        # colors
        'axes.prop_cycle': cycler('color', get_palette(dict=False, hex=True)),

        # grid
        'axes.facecolor': 'white',
        'axes.edgecolor': '.8',
        'axes.grid': False,
        'grid.linestyle': '-',
        'grid.linewidth': 0.25,
        'grid.color': '#a3a3a3',
        'axes.linewidth': 0.0,
    }

    rc_params.update(other_rc_params)

    def _adjust_figure_inplace(fig, ax):
            # set just x axis grid lines
            ax.grid(True)
            ax.xaxis.grid(False)

            if not y_axis_grid:
                ax.yaxis.grid(False)

            ax.set_axisbelow(True)

            if zero_lines:
                ax.axvline(x=0, c='k', linestyle='-', lw=0.7, alpha=0.5)
                ax.axhline(y=0, c='k', linestyle='-', lw=0.7, alpha=0.5)

            if x_formatter:
                ax.xaxis.set_major_formatter(x_formatter)
            if y_formatter:
                ax.yaxis.set_major_formatter(y_formatter)

            fig.tight_layout()

    if seaborn:
        with sns.axes_style(rc=rc_params):
            sns.set_palette(get_palette())
            yield
            fig = plt.gcf()
            fig.set_size_inches(figure_size)

            # hack for joint plots (looks like main plot is always first)
            ax = plt.gcf().get_axes()[0]

            _adjust_figure_inplace(fig, ax)
    else:
        with plt.rc_context(rc_params):
            if xor((subplots_rows is not None), (subplots_columns is not None)):
                raise ValueError("Must pass both subplots_rows and subplots_columns or neither.")

            if subplots_rows is not None:
                fig, axes = plt.subplots(subplots_rows,
                                         subplots_columns,
                                         figsize=figure_size,
                                         **subplots_kwargs)

                for ax in axes:
                    _adjust_figure_inplace(fig, ax)

                yield axes
            else:
                fig, ax = plt.subplots(figsize=figure_size)

                _adjust_figure_inplace(fig, ax)
                yield ax

        # reset after context manager is closed
        for k, v in original_params.items():
            plt.rcParams[k] = v


def publish_image(path, transparent=False):
    plt.tight_layout()

    dir_path = os.path.dirname(path)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    plt.savefig(path, bbox_inches='tight', transparent=transparent)


def scaleless_round(x, precision=1):
    """ Rounds to `precision` decimal places IN SCIENTIFIC
        NOTATION (so, regardless of scale).

        E.g, scaleless_round(1100, 1) = 1.1e2
             scaleless_round(15000, 1) = 1.5e3
    """
    def _rounder(x, precision=precision):
        if isinstance(x, pd.Series) and x.dtype == 'object':
            return x
        elif isinstance(x, str):
            return x

        f_str = '{{:.{}e}}'.format(precision)

        digits, degree = list(map(float, f_str.format(float(x)).split('e')))

        return float(digits * (10**degree))

    if isinstance(x, pd.Series):
        return x.apply(_rounder)
    elif isinstance(x, pd.DataFrame):
        return x.applymap(_rounder)
    else:
        return _rounder(x)


def to_pct(x):
    """ Converts to percent. Columnwise pcts if x is a DataFrame.
    """
    series_to_pct = lambda z: z.divide(z.sum()) * 100 if isinstance(z, pd.Series) and z.dtype != 'object' else z

    if isinstance(x, pd.Series):
        return series_to_pct(x)
    elif isinstance(x, pd.DataFrame):
        return x.apply(series_to_pct, axis=0)
    else:
        total = sum(x)
        return [100 * float(z) / float(total) for z in x]


def add_noise(x, scale=10):
    def _series_add_noise(z):
        if isinstance(z, pd.Series) and z.dtype == 'object':
            return z

        noised = np.random.normal(0, np.std(z), size=z.shape) / scale + z

        if (z >= 0).all():
            return np.abs(noised)
        else:
            return noised

    if isinstance(x, pd.Series):
        return _series_add_noise(x)
    elif isinstance(x, pd.DataFrame):
        return x.apply(_series_add_noise, axis=0)
    else:
        return _series_add_noise(np.array(x))


def get_palette(hex=False, dict=False):
    """ Returns the palette.
            With names if dict=True (else a list)
            Hex vals if hex=True (else a matplotlib rgb color)
    """
    palette = PALETTE

    if not hex:
        palette = OrderedDict([(k,  colors.ColorConverter().to_rgb(v)) for k, v in PALETTE.items()])

    if not dict:
        palette = palette.values()

    return palette


def get_semantic_color(concept, default=None):
    mapping = {
        'urban': get_palette(dict=True)['pink'],
        'rural': get_palette(dict=True)['green'],
        'cash in': get_palette(dict=True)['purple'],
        'cash out': get_palette(dict=True)['red'],
        'c2c transfer': get_palette(dict=True)['yellow'],
        'none': get_palette(dict=True)['grey'],
    }

    color = mapping.get(concept.lower().replace('-', ' ').strip(), None)

    if color is not None:
        return color
    else:
        if default is not None:
            return get_palette(dict=True)[default]
        else:
            error_format = "Semantic colors not defined for '{}'; valid values are '{}'."
            raise ValueError(error_format.format(concept, mapping.keys()))


def comma_func_formatter():
    def _comma_formatter(x, pos):
        return '{:,.0f}'.format(x)
    return FuncFormatter(_comma_formatter)


def pct_func_formatter(already_scaled=False):
    def _pct_formatter(x, pos):
        x = x if already_scaled else 100 * x
        return '{:.0f}%'.format(x)
    return FuncFormatter(_pct_formatter)


def donut(ax, sizes, text, colors, labels, label_counts=False, label_percentages=False, explode=None):
    """ It's donut time!

        :param ax: The axis to plot on.
        :param sizes: The raw values (not percentages) for the size of the segments.
                      NOTE: Should be sorted in ASCENDING order.
        :param text: The text in the center of the donut.
        :param colors: The colors for the segment. If none, it is generated with Viridis.
        :param labels: The labels for the segments.
        :param label_counts: Wheter or not to append the counts. (Can be a formatting function.)
        :param label_perecentages: Whether or not to append the percentage.
        :param explode: A float value for separating segments (0.3 works well)
    """
    width = 0.35

    if colors is None:
        colors = sns.color_palette("viridis_r", len(labels))

    if label_counts:
        if callable(label_counts):
            counts = [label_counts(s) for s in sizes]
        else:
            counts = sizes

        labels = ['{} [{}]'.format(l, counts[i]) for i, l in enumerate(labels)]

    if label_percentages:
        total = sum(sizes)
        labels = ['{} ({:.0f}%)'.format(l, (100 * sizes[i]) / total) for i, l in enumerate(labels)]

    kwargs = dict(colors=colors, startangle=0)

    if explode is not None:
        kwargs['explode'] = [explode] * len(labels)

    outside, _ = ax.pie(sizes,
                        radius=1,
                        pctdistance=1 - width / 2,
                        labels=labels,
                        **kwargs)

    plt.setp(outside, width=width, edgecolor='white')

    kwargs = dict(size=20, fontweight='bold', va='center')
    ax.text(0, 0, text, ha='center', **kwargs)


def labeled_bar(ax, to_plot, percentages=False, label_formatter=None, horizontal=False, font_size=12, **kwargs):
    x_text_offset, y_text_offset = 0, 0
    if horizontal:
        ax = to_plot.plot.barh(ax=ax, **kwargs)
        ax.set_xticklabels([])
        x_text_offset = 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])
    else:
        ax = to_plot.plot.bar(ax=ax, **kwargs)
        ax.set_yticklabels([])
        y_text_offset = 0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0])

    for s in ax.spines:
        ax.spines[s].set_visible(False)

    ax.tick_params(axis='both', which='both', length=0)

    if percentages:
        to_plot = to_pct(to_plot)

    for i, v in enumerate(to_plot.values.tolist()):
        # handle stacked charts
        if isinstance(v, list):
            v = sum(v)

        v_str = label_formatter(v) if label_formatter else str(v)
        ax.text(v + x_text_offset, i + y_text_offset, v_str, dict(size=font_size), va='center')

    return ax
