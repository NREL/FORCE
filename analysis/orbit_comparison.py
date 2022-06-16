__author__ = ["Jake Nunemaker"]
__copyright__ = "Copyright 2021, National Renewable Energy Laboratory"
__maintainer__ = "Jake Nunemaker"
__email__ = "jake.nunemaker@nrel.gov"


import os
import numpy as np
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt
from ORBIT import ProjectManager, load_config
from ORBIT.core.library import initialize_library
from FORCE.learning import Regression
from plot_routines import scatter_plot, plot_forecast, plot_forecast_comp, plot_deployment
import pprint as pp


DIR = os.path.split(__file__)[0]
LIBRARY = os.path.join(DIR, "library")
initialize_library(LIBRARY)


# TODO: Reindex all data to the same starting year before anything happens.
# TODO: May need to revise how forecasts are input

# Create results folder if it doesn't exist
results_dirs = [
    os.path.join(DIR, "results", "statistics")
]

for d in results_dirs:
    if not os.path.exists(d):
        os.makedirs(d)

### Initialize Data
scenario = 'high cost'
if scenario == 'baseline':
    capacity = 'capacity'
    opex_scale = 1
    ncf_scale = 1
    fig_label = 'baseline_deploy'
elif scenario == 'high':
    capacity = 'high capacity'
    opex_scale = 1
    ncf_scale = 1
    fig_label = 'high_deploy'
elif scenario == 'low':
    capacity = 'low capacity'
    opex_scale = 1
    ncf_scale = 1
    fig_label = 'low_deploy'
elif scenario == 'low cost':
    capacity = 'capacity'
    fig_label = 'low_cost'
    opex_scale = 0.9
    ncf_scale = 1.05
elif scenario == 'high cost':
    capacity = 'capacity'
    fig_label = 'high_cost'
    opex_scale = 1.1
    ncf_scale = 0.95

## Forecast
FORECAST_FP_FIXED = os.path.join(DIR, "data", "2021_fixed_forecast.csv")
FORECAST_FIXED = pd.read_csv(FORECAST_FP_FIXED).set_index("year").to_dict()[capacity]
FORECAST_FP_FLOATING = os.path.join(DIR, "data", "2021_floating_forecast.csv")
FORECAST_FLOATING = pd.read_csv(FORECAST_FP_FLOATING).set_index("year").to_dict()[capacity]

plot_deployment(list(FORECAST_FIXED.keys()), list(FORECAST_FIXED.values()),
                list(FORECAST_FLOATING.keys()), list(FORECAST_FLOATING.values()),
                'results/deployment.png'
                )

## Scaling factor for demonstration-scale floating capex
FLOATING_DEMO_SCALE = 2.5     # Set initial Capex around $10K/kw
FLOATING_CAPACITY_2020 = 91   # Cumulative capacity as of previous year.  From OWMR.

## Regression Settings
PROJECTS = pd.read_csv(os.path.join(DIR, "data", "2021_OWMR.csv"), header=2)
FILTERS = {
    'Capacity MW (Max)': (149, ),
    'Full Commissioning': (2014, 2021),
}
TO_AGGREGATE = {
    'United Kingdom': 'United Kingdom',
    'Germany': 'Germany',
    'Netherlands': 'Netherlands',
    'Belgium' : 'Belgium',
    'China': 'China',
    'Denmark': 'Denmark',
}
TO_DROP = []
FIXED_PREDICTORS = [
            'Country Name',
            'Water Depth Max (m)',
            # 'Turbine MW (Max)',
            'Capacity MW (Max)',
            'Distance From Shore Auto (km)',
            ]
FLOAT_PREDICTORS = [
            'Country Name',
            'Water Depth Max (m)',
            # 'Turbine MW (Max)',
            # 'Capacity MW (Max)',
            'Distance From Shore Auto (km)',
            ]


## ORBIT Sites + Configs
ORBIT_FIXED_SITES = {
    "Site 1": {
        2021: "site_1_2021.yaml",
        2025: "site_1_2025.yaml",
        2030: "site_1_2030.yaml",
        2035: "site_1_2035.yaml"
    },

    "Site 2": {
        2021: "site_2_2021.yaml",
        2025: "site_2_2025.yaml",
        2030: "site_2_2030.yaml",
        2035: "site_2_2035.yaml"
    },

    "Site 3": {
        2021: "site_3_2021.yaml",
        2025: "site_3_2025.yaml",
        2030: "site_3_2030.yaml",
        2035: "site_3_2035.yaml"
    },

    "Site 4": {
        2021: "site_4_2021.yaml",
        2025: "site_4_2025.yaml",
        2030: "site_4_2030.yaml",
        2035: "site_4_2035.yaml"
    },

    "Site 5": {
        2021: "site_5_2021.yaml",
        2025: "site_5_2025.yaml",
        2030: "site_5_2030.yaml",
        2035: "site_5_2035.yaml"
    }
}

ORBIT_FLOATING_SITES = {
    "Site 1": {
        2021: "site_1_2021.yaml",
        # 2025: "site_1_2025.yaml",
        # 2030: "site_1_2030.yaml",
        2035: "site_1_2035.yaml"
    },

    "Site 2": {
        2021: "site_1_2021.yaml",
        # 2025: "site_1_2025.yaml",
        # 2030: "site_1_2030.yaml",
        2035: "site_1_2035.yaml"
    },

    "Site 3": {
        2021: "site_3_2021.yaml",
        # 2025: "site_1_2025.yaml",
        # 2030: "site_1_2030.yaml",
        2035: "site_3_2035.yaml"
    },

    "Site 4": {
        2021: "site_4_2021.yaml",
        # 2025: "site_1_2025.yaml",
        # 2030: "site_1_2030.yaml",
        2035: "site_4_2035.yaml"
    },

    "Site 5": {
        2021: "site_5_2021.yaml",
        # 2025: "site_1_2025.yaml",
        # 2030: "site_1_2030.yaml",
        2035: "site_5_2035.yaml"
    },
}

### Functions
def run_regression(projects, filters, to_aggregate, to_drop, predictors):
    """
    Run FORCE Regression with given settings.

    Parameters
    ----------
    projects : DataFrame
    filters : dict
    to_aggregate : dict
    to_drop : list
        List of countries to drop.
    """

    regression = Regression(
        projects,
        y_var="log CAPEX_per_kw",
        filters=filters,
        regression_variables=predictors,
        aggregate_countries=to_aggregate,
        drop_categorical=["United Kingdom"],
        drop_country=to_drop,
        log_vars=['Cumulative Capacity', 'CAPEX_per_kw'],
    )
    print(regression.summary)
    return regression

def stats_check(regression):
    summary_stats = {'R2': regression.r2,
                     'Adjusted R2': regression.r2_adj,
                     'Experience factor': regression.cumulative_capacity_fit,
                     'Experience factor standard error': regression.cumulative_capacity_bse,
                     'Learning rate': regression.learning_rate,
                     }
    predictor_stats = zip(regression.params_dict.values(),
                          regression.pvalues.keys(),
                          regression.pvalues.values,
                          regression.vif)

    # Write stats results to Excel
    xlsfile = "results/statistics/stats_output.xlsx"
    workbook = xlsxwriter.Workbook(xlsfile)
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0

    # Write all data to workbook.  Start with scalars then per-predictor values
    for k, v in summary_stats.items():
        worksheet.write(row, col, k)
        worksheet.write(row, col+1, v)
        row+=1
    row+=1
    worksheet.write(row, col, 'Predictor variable')
    worksheet.write(row, col + 1, 'Coefficient')
    worksheet.write(row, col+2, 'P-value')
    worksheet.write(row, col+3, 'VIF')
    row+=1
    for b, var, p, v in predictor_stats:
        worksheet.write(row, col, var)
        worksheet.write(row, col + 1, b)
        worksheet.write(row, col+2, p)
        worksheet.write(row, col+3, v)
        row+=1
    workbook.close()

    # Plot residuals
    res_x = regression.fittedvalues
    res_y = regression.residuals

    return res_x, res_y


def linearize_forecast(forecast):
    """
    Linearize the forecasted capacity over forecast period.

    Parameters
    ----------
    forecast : dict
    """

    years = np.arange(min(forecast.keys()), max(forecast.keys()) + 1)
    linear = np.linspace(min(forecast.values()), max(forecast.values()), len(years))
    f2 = {k: linear[i] for i, k in enumerate(years)}

    return years, f2


def _zip_into_years(start, stop, years):
    return {yr: val for yr, val in zip(years, np.linspace(start, stop, len(years)))}


def run_orbit_configs(sites, b0, upcoming, years, opex_scale, ncf_scale, initial_capex=None, fixfloat='fixed'):
    """"""

    orbit_outputs = []
    for name, configs in sites.items():

        site_data = pd.DataFrame(index=years)

        for yr, c in configs.items():

            config = load_config(os.path.join(DIR, "orbit_configs", fixfloat, c))
            weather_file = config.pop("weather", None)

            if weather_file is not None:
                weather = pd.read_csv(os.path.join(DIR, "library", "weather", weather_file)).set_index("datetime")

            else:
                weather = None

            #TODO: better indexing
            if yr == 2021:
                ncf_i = config['project_parameters']['ncf']
                opex_i = config['project_parameters']['opex']
                fcr_i = config['project_parameters']['fcr']
            elif yr == 2035:
                ncf_f = config['project_parameters']['ncf']
                opex_f = config['project_parameters']['opex']
                fcr_f = config['project_parameters']['fcr']

            project = ProjectManager(config, weather)
            project.run()

            if fixfloat == 'fixed':
                site_data.loc[int(yr), "ORBIT"] = project.total_capex_per_kw
            else:
                # Scale floating Capex to demo scale projects
                site_data.loc[int(yr), "ORBIT"] = project.total_capex_per_kw * FLOATING_DEMO_SCALE


        min_yr = min(configs.keys())  # TODO: What if min_yr doesn't line up with first forecast year?

        # TODO: Not sure which one of these is more appropriate.
        # c = site_data.loc[min_yr, "ORBIT"] / (regression.installed_capacity ** b0)
        if initial_capex:
            c = initial_capex / (upcoming[min_yr] ** b0)
        else:
            c = site_data.loc[min_yr, "ORBIT"] / (upcoming[min_yr] ** b0)
        site_data.loc[min_yr, "Regression"] = c * upcoming[min_yr] ** b0
        for yr in years[1:]:
            site_data.loc[yr, "Regression"] = c * upcoming[yr] ** b0

        # Define Opex, NCF, FCR arrays
        OPEX = (opex_i, opex_scale * opex_f)
        NCF = (ncf_i, ncf_scale * ncf_f)
        FCR = (fcr_i, fcr_f)
        opex = {yr: val for yr, val in zip(years, np.linspace(*OPEX, len(years)))}
        ncf = {yr: val for yr, val in zip(years, np.linspace(*NCF, len(years)))}
        fcr = {yr: val for yr, val in zip(years, np.linspace(*FCR, len(years)))}

        site_data["OpEx"] = opex.values()
        aep = {k: v * 8760 for k, v in ncf.items()}  # MWh
        site_data["AEP"] = aep.values()
        site_data["FCR"] = fcr.values()
        site_data["LCOE"] = 1000 * (site_data["FCR"] * site_data["Regression"] + site_data["OpEx"]) / site_data["AEP"]
        site_data["Site"] = name

        orbit_outputs.append(site_data)

    combined_outputs = pd.concat(orbit_outputs)

    return combined_outputs

def regression_and_plot(FORECAST, PROJECTS, FILTERS, TO_AGGREGATE, TO_DROP, PREDICTORS, ORBIT_SITES,
                        fixfloat='fixed', opex_scale=1, ncf_scale=1):
    """Run all subroutines to create regression fit and all plots"""
    # Forecast
    years, linear_forecast = linearize_forecast(FORECAST)

    # Regression
    regression = run_regression(PROJECTS, FILTERS, TO_AGGREGATE, TO_DROP, PREDICTORS)
    res_x, res_y = stats_check(regression)
    b0 = regression.cumulative_capacity_fit
    bse = regression.cumulative_capacity_bse
    if fixfloat == 'fixed':
        # Todo: Should this also be FORECAST instead of linear_forecast?
        upcoming_capacity = {
            k: v - regression.installed_capacity for k, v in linear_forecast.items()
        }
    else:
        # Todo: Move to Regression class
        upcoming_capacity = {
            k: v - FLOATING_CAPACITY_2020 for k, v in FORECAST.items()
        }

    # ORBIT Results
    combined_outputs = run_orbit_configs(ORBIT_SITES, b0, upcoming_capacity, years,
                                         opex_scale, ncf_scale, fixfloat=fixfloat)
    initial_capex_range = combined_outputs.loc[2021, 'ORBIT'].values
    avg_start = pd.pivot_table(combined_outputs.reset_index(), values='ORBIT', index='index').iloc[0].values[0]
    std_start =  \
        pd.pivot_table(combined_outputs.reset_index(), values='ORBIT', index='index', aggfunc=np.std).iloc[0].values[0]
    std_lcoe_start = \
        pd.pivot_table(combined_outputs.reset_index(), values='LCOE', index='index', aggfunc=np.std).iloc[0].values[0]

    # Bounds for faster/slower learning rate
    combined_outputs_max_conservative = run_orbit_configs(ORBIT_SITES, b0 + bse, upcoming_capacity, years,
                                                      opex_scale, ncf_scale, initial_capex=avg_start + std_start,
                                                      fixfloat=fixfloat)
    combined_outputs_min_aggressive = run_orbit_configs(ORBIT_SITES, b0 - bse, upcoming_capacity, years,
                                                    opex_scale, ncf_scale, initial_capex=avg_start - std_start,
                                                    fixfloat=fixfloat)
    combined_outputs_avg_conservative = run_orbit_configs(ORBIT_SITES, b0 + bse, upcoming_capacity, years,
                                                          opex_scale, ncf_scale, fixfloat=fixfloat)
    combined_outputs_avg_aggressive = run_orbit_configs(ORBIT_SITES, b0 - bse, upcoming_capacity, years,
                                                        opex_scale, ncf_scale, fixfloat=fixfloat)

    # Capex
    avg_capex = np.array(pd.pivot_table(combined_outputs.reset_index(),
                                        values='Regression', index='index').loc[:,'Regression'])
    avg_capex_conservative = \
        np.array(pd.pivot_table(combined_outputs_avg_conservative.reset_index(),
                                values='Regression', index='index').loc[:, 'Regression'])
    avg_capex_aggressive = \
        np.array(pd.pivot_table(combined_outputs_avg_aggressive.reset_index(),
                                values='Regression', index='index').loc[:, 'Regression'])
    max_capex_conservative = \
        np.array(pd.pivot_table(combined_outputs_max_conservative.reset_index(),
                                values='Regression', index='index', aggfunc=max).loc[:,'Regression'])
    min_capex_aggressive = \
        np.array(pd.pivot_table(combined_outputs_min_aggressive.reset_index(),
                                values='Regression', index='index', aggfunc=min).loc[:,'Regression'])
    # print('Avg', avg_capex)
    # print('Avg cons', avg_capex_conservative)
    # print('Avg agg', avg_capex_aggressive)
    # print('Max cons', max_capex_conservative)
    # print('Min agg', min_capex_aggressive)

    # LCOE
    avg_lcoe = np.array(pd.pivot_table(combined_outputs.reset_index(), values='LCOE', index='index').loc[:,'LCOE'])
    max_lcoe_conservative = np.array(pd.pivot_table(combined_outputs_max_conservative.reset_index(),
                                                    values='LCOE', index='index', aggfunc=max).loc[:,'LCOE'])
    min_lcoe_aggressive = np.array(pd.pivot_table(combined_outputs_min_aggressive.reset_index(),
                                                  values='LCOE', index='index', aggfunc=min).loc[:,'LCOE'])

    # print('Avg', avg_lcoe)
    # print('Max cons', max_lcoe_conservative)
    # print('Min agg', min_lcoe_aggressive)

    ### Write data
    pd.DataFrame({
        'Year': years,
        'Capex': avg_capex,
        'Capex (average start and conservative LR)': avg_capex_conservative,
        'Capex (average start and aggressive LR)': avg_capex_aggressive,
        'Capex (max start and conservative LR)': max_capex_conservative,
        'Capex (min start and aggressive LR)': min_capex_aggressive,
        'Capex percent reductions': 1 - avg_capex / avg_capex[0],
        'LCOE': avg_lcoe,
        'LCOE (max start and conservative LR)': max_lcoe_conservative,
        'LCOE (min start and aggressive LR)': min_lcoe_aggressive,
        'LCOE percent reductions': 1 - avg_lcoe/ avg_lcoe[0]
    }).to_csv('results/' + fixfloat + '_' + fig_label + '_data_out.csv')


    ### Plotting
    # Forecast
    fname_capex = 'results/' + fixfloat + '_capex_forecast_' + fig_label + '.png'
    fig_capex, ax_capex = plot_forecast(
        upcoming_capacity,
        avg_capex,
        min_capex_aggressive,
        max_capex_conservative,
        std_start,
        ylabel='CapEx, $/kW',
        fixfloat=fixfloat,
        fname=fname_capex
    )

    fname_lcoe = 'results/' + fixfloat + '_lcoe_forecast_' + fig_label + '.png'
    plot_forecast(
        upcoming_capacity,
        avg_lcoe,
        min_lcoe_aggressive,
        max_lcoe_conservative,
        std_lcoe_start,
        ylabel='LCOE, $/MWh',
        fixfloat=fixfloat,
        fname=fname_lcoe
    )

    # Residuals
    fname_residuals = 'results/statistics/' + fixfloat + '_residuals.png'
    scatter_plot(res_x, res_y, 'Fitted values (log of CapEx)', 'Residuals', fname=fname_residuals)

    # Sensitivities
    fname_uncert = 'results/' + fixfloat + '_capex_forecast' + fig_label + '_uncertainty.png'
    plot_forecast_comp(
        fig_capex,
        ax_capex,
        upcoming_capacity,
        avg_capex,
        avg_capex_aggressive,
        avg_capex_conservative,
        initial_capex_range,
        ylabel='CapEx, $/kW',
        fname=fname_uncert
    )



### Main Script
if __name__ == "__main__":

    # Fixed bottom
    regression_and_plot(FORECAST_FIXED, PROJECTS, FILTERS, TO_AGGREGATE, TO_DROP, FIXED_PREDICTORS, ORBIT_FIXED_SITES,
                        opex_scale=opex_scale, ncf_scale=ncf_scale)

    # Floating
    regression_and_plot(FORECAST_FLOATING, PROJECTS, FILTERS, TO_AGGREGATE, TO_DROP, FLOAT_PREDICTORS, ORBIT_FLOATING_SITES,
                        fixfloat='floating', opex_scale=opex_scale, ncf_scale=ncf_scale)
