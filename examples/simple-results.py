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


# Create results folder if it doesn't exist
results_dirs = [
    os.path.join(DIR, "results", "statistics")
]

for d in results_dirs:
    if not os.path.exists(d):
        os.makedirs(d)

# Define start and end years
start_year = 2021
end_year = 2035

# Optional: Define scaling factors for Opex and NCF in the final year (e.g., Opex in end_year = Opex in start_year * opex_scale). Can also control this directly in the end_year config files.
opex_scale = 1
ncf_scale = 1

## Select CSV files for fixed bottom and floating deployment
FORECAST_FP_FIXED = os.path.join(DIR, "data", "2021_fixed_forecast.csv")
FORECAST_FIXED = pd.read_csv(FORECAST_FP_FIXED).set_index("year").to_dict()['capacity']
FORECAST_FP_FLOATING = os.path.join(DIR, "data", "2021_floating_forecast.csv")
FORECAST_FLOATING = pd.read_csv(FORECAST_FP_FLOATING).set_index("year").to_dict()['capacity']

# Optional - plot the deployment trajectory
plot_deployment(list(FORECAST_FIXED.keys()), list(FORECAST_FIXED.values()),
                list(FORECAST_FLOATING.keys()), list(FORECAST_FLOATING.values()),
                'results/deployment.png'
                )

## Scaling factor for demonstration-scale floating capex
FLOATING_DEMO_SCALE = 2.5     # Set initial Capex around $10K/kw
FLOATING_CAPACITY_2020 = 91   # Cumulative capacity as of previous year.  From OWMR.

## Use historic project data to compute a learning rate. File needs to be use the same format as data/project_list_template.csv. NREL uses the 4C Offshore Database to populate this file
PROJECTS = pd.read_csv(os.path.join(DIR, "data", "project_list_template.csv"), header=2)

# Define any filters for the data set. In this case, we filter out projects below 150 MW and projects commissioned before 2014
FILTERS = {
    'Capacity MW (Max)': (149, ),
    'Full Commissioning': (2014, start_year),
}
# Optional step to aggregate countries as control variables in the regression. For example, could do {'Germany': 'Northern Europe', 'Belgium': 'Northern Europe'}, etc.

TO_AGGREGATE = {
    'United Kingdom': 'United Kingdom',
    'Germany': 'Germany',
    'Netherlands': 'Netherlands',
    'Belgium' : 'Belgium',
    'China': 'China',
    'Denmark': 'Denmark',
}
# Optional step to drop out countries from the regression analysis.
TO_DROP = []
# Define the predictor variables to use in the regression. Any of the columns from the input CSV ('data/project_list_template.csv') are valid. Predictor variables control for these effects in the regression - in other words, the learning rate is independent from changes in the predictor.
FIXED_PREDICTORS = [
            'Country Name',
            'Water Depth Max (m)',
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


## Define fixed bottom and floating sites to calculate initial capex. Also define future sites, althoguh capex isn't used. ORBIT configs also contain values for Opex, NCF, FCR per year.
# By defining a variety of different sites, FORCE estimates the initial variance in CapEx due to site conditions.
ORBIT_FIXED_SITES = {
    "Site 1": {
        start_year: "site_1_2021.yaml",
        end_year: "site_1_2035.yaml"
    },

    "Site 2": {
        start_year: "site_2_2021.yaml",
        end_year: "site_2_2035.yaml"
    },

    "Site 3": {
        start_year: "site_3_2021.yaml",
        end_year: "site_3_2035.yaml"
    },

    "Site 4": {
        start_year: "site_4_2021.yaml",
        end_year: "site_4_2035.yaml"
    },

    "Site 5": {
        start_year: "site_5_2021.yaml",
        end_year: "site_5_2035.yaml"
    }
}

ORBIT_FLOATING_SITES = {
    "Site 1": {
        start_year: "site_1_2021.yaml",
        end_year: "site_1_2035.yaml"
    },

    "Site 2": {
        start_year: "site_1_2021.yaml",
        end_year: "site_1_2035.yaml"
    },

    "Site 3": {
        start_year: "site_3_2021.yaml",
        end_year: "site_3_2035.yaml"
    },

    "Site 4": {
        start_year: "site_4_2021.yaml",
        end_year: "site_4_2035.yaml"
    },

    "Site 5": {
        start_year: "site_5_2021.yaml",
        end_year: "site_5_2035.yaml"
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
    Linearize the forecasted capacity over forecast period. Thsi helps to smooth out the projected costs.

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
            if yr == start_year:
                ncf_i = config['project_parameters']['ncf']
                opex_i = config['project_parameters']['opex']
                fcr_i = config['project_parameters']['fcr']
            elif yr == end_year:
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


        min_yr = min(configs.keys())

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

    # Run the regression analysis to compute the learning rate
    regression = run_regression(PROJECTS, FILTERS, TO_AGGREGATE, TO_DROP, PREDICTORS)
    # Check the residuals of the regression
    res_x, res_y = stats_check(regression)
    # Extract key variables from the regression fit (the learning rate and the standard error)
    b0 = regression.cumulative_capacity_fit
    bse = regression.cumulative_capacity_bse

    # Extract the upcoming capacity per year in the forecast to develop the cost forecast
    if fixfloat == 'fixed':
        upcoming_capacity = {
            k: v - regression.installed_capacity for k, v in linear_forecast.items()
        }
    else:
        # Todo: Move to Regression class
        upcoming_capacity = {
            k: v - FLOATING_CAPACITY_2020 for k, v in FORECAST.items()
        }

    # Run ORBIT to define initial site costs. Prescribe initial and final values for Opex, NCF, FCR
    combined_outputs = run_orbit_configs(ORBIT_SITES, b0, upcoming_capacity, years, opex_scale, ncf_scale, fixfloat=fixfloat)
    # Save the initial capex values for all defined sites for plotting

    initial_capex_range = combined_outputs.loc[start_year, 'ORBIT'].values
    # Define the average starting capex and the standard deviation between all sites
    avg_start = pd.pivot_table(combined_outputs.reset_index(), values='ORBIT', index='index').iloc[0].values[0]
    std_start =  \
        pd.pivot_table(combined_outputs.reset_index(), values='ORBIT', index='index', aggfunc=np.std).iloc[0].values[0]

    # Use high/low values of learning rates (b0 +/- bse) to define different cost trajectories and subsequent uncertainty
    # Naming convention:
    #   _max_conservative: Maximum initial capex, conservative learning rate (highest capex in end year)
    #   _min_aggressive: Minimum initial capex, aggressive learning rate (lowest capex in end year)
    combined_outputs_max_conservative = run_orbit_configs(ORBIT_SITES, b0 + bse, upcoming_capacity, years, opex_scale, ncf_scale, initial_capex=avg_start + std_start,fixfloat=fixfloat)
    combined_outputs_min_aggressive = run_orbit_configs(ORBIT_SITES, b0 - bse, upcoming_capacity, years,opex_scale, ncf_scale, initial_capex=avg_start - std_start,fixfloat=fixfloat)

    # Create an array of capex trajectories
    avg_capex = np.array(pd.pivot_table(combined_outputs.reset_index(),
                                        values='Regression', index='index').loc[:,'Regression'])
    max_capex_conservative = \
        np.array(pd.pivot_table(combined_outputs_max_conservative.reset_index(),
                                values='Regression', index='index', aggfunc=max).loc[:,'Regression'])
    min_capex_aggressive = \
        np.array(pd.pivot_table(combined_outputs_min_aggressive.reset_index(),
                                values='Regression', index='index', aggfunc=min).loc[:,'Regression'])

    ### Write data
    pd.DataFrame({
        'Year': years,
        'Capex': avg_capex,
        'Capex (max start and conservative LR)': max_capex_conservative,
        'Capex (min start and aggressive LR)': min_capex_aggressive,
        'Capex percent reductions': 1 - avg_capex / avg_capex[0],
    }).to_csv('results/' + fixfloat + '_data_out.csv')


    ### Plotting
    # Forecast
    fname_capex = 'results/' + fixfloat + '_capex_forecast.png'
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

    # Residuals
    fname_residuals = 'results/statistics/' + fixfloat + '_residuals.png'
    scatter_plot(res_x, res_y, 'Fitted values (log of CapEx)', 'Residuals', fname=fname_residuals)


### Main Script
if __name__ == "__main__":

    # Fixed bottom
    regression_and_plot(FORECAST_FIXED, PROJECTS, FILTERS, TO_AGGREGATE, TO_DROP, FIXED_PREDICTORS, ORBIT_FIXED_SITES,
                        opex_scale=opex_scale, ncf_scale=ncf_scale)

    # Floating
    regression_and_plot(FORECAST_FLOATING, PROJECTS, FILTERS, TO_AGGREGATE, TO_DROP, FLOAT_PREDICTORS, ORBIT_FLOATING_SITES,
                        fixfloat='floating', opex_scale=opex_scale, ncf_scale=ncf_scale)
