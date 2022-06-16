__author__ = ["Jake Nunemaker", "Matt Shields", "Philipp Beiter"]
__copyright__ = "Copyright 2022, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"
__status__ = "Development"


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_learning_forecast(
    installed,
    capex,
    fit,
    forecast,
    bse=None,
    axes=None,
    perc_change=False,
    data_file=None,
    **kwargs,
):
    """
    Plots forecasted CAPEX/kW based on the installed capacity, current capex,
    fit parameters and the forecasted cumulative capacity.

    Parameters
    ----------
    installed : float
        Installed capacity at start of forecast (MW).
    capex: float
        CAPEX at start of forecast ($/kW)
    fit : float
    forecast : dict
        Dictionary of forecasted capacity with format:
        'year': 'MW of capacity'.
    bse : float | None
        Standard error of the fit.
        If None, error will not be plotted.
    axes : matplotlib.Axis
    perc_change : bool
    data_file :
    """

    if axes is None:
        fig = plt.figure(**kwargs)
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()

    else:
        raise NotImplementedError(
            "Passing in pre-constructed axes is not supported yet."
        )

    upcoming = [v - installed for _, v in forecast.items()]

    x = np.linspace(installed, upcoming[-1])
    b0 = fit
    C0_0 = capex / (installed ** b0)
    # # y0 = C0_0 * x ** b0
    # y0 = 1 - C0_0 * x ** b0 / capex
    # y_per_year = 1 - C0_0 * upcoming ** b0 / capex

    if perc_change is False:
        y0 = calc_curve(x, C0_0, b0)
        y0_per_year = calc_curve(upcoming, C0_0, b0)
        _out_col = "Average global CapEx, $/KW"

    else:
        y0 = calc_curve(x, C0_0, b0, capex_0=capex)
        y0_per_year = calc_curve(upcoming, C0_0, b0, capex_0=capex)
        _out_col = "Percent change from initial CapEx"

    ax1.plot(x, y0, "k-")
    ax1.set_xlabel("Cumulative Capacity")
    ax1.set_ylabel("CAPEX, $/KW")

    if bse:
        b1 = fit + bse
        b2 = fit - bse

        C0_1 = capex / (installed ** b1)
        C0_2 = capex / (installed ** b2)

        # y1 = C0_1 * x ** b1
        # y2 = C0_2 * x ** b2
        if perc_change is False:
            y1 = calc_curve(x, C0_1, b1)
            y2 = calc_curve(x, C0_2, b2)
        else:
            y1 = calc_curve(x, C0_1, b1, capex_0=capex)
            y2 = calc_curve(x, C0_2, b2, capex_0=capex)
        ax1.fill_between(x, y1, y2)

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(upcoming)
    ax2.set_xticklabels(forecast.keys(), rotation=45, fontsize=8)
    ax2.set_ylabel("Projected COD")

    if data_file:
        _out = pd.DataFrame({"Year": forecast.keys(), _out_col: y0_per_year})

        _out.set_index("Year").to_csv(data_file)

    return ax1, ax2


def calc_curve(x, C0, b, capex_0=None):
    """Fit the learning curve to a prescribed range of years"""
    if capex_0:
        """Determine percent change from initial capex value"""
        y = 1 - C0 * x ** b / capex_0
    else:
        y = C0 * x ** b

    return y
