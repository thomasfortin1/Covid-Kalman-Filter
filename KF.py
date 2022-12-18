import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import geopandas as gpd

baseline = 2.69
masking = 2.69 * .81
lockdown = .7

state_name_to_abbr = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
}

abbr_to_state = {v: k for k, v in state_name_to_abbr.items()}


def get_state_name_to_abbr():
    return state_name_to_abbr


def kalman_filter_predict(z, P, A, B):
    # Predict
    z_hat = A @ z
    P_hat = A @ P @ A.T + B @ B.T
    return z_hat, P_hat


def kalman_filter_update(m, z_hat, P_hat, C, R):
    # Update
    K = P_hat @ C.T @ np.linalg.inv(C @ P_hat @ C.T + R)
    z = z_hat + K @ (m - C @ z_hat)
    P = P_hat @ C.T * np.linalg.inv(C @ P_hat @ C.T + R)
    return z, P


def get_state_list():
    state = gpd.read_file('../CovIDSpatialShifts/COUNTY_dis2/COUNTY_dis2.shp')
    return state['STATE_NAME'].values


def run_individual_test(region, make_plot=False):
    cov_df = pd.read_csv('Weekly_United_States_COVID-19_Cases_and_Deaths_by_State (1).csv',
                         dtype={'State': str, 'start_date': str, 'end_date': str,
                                'tot_cases': int, 'new_cases': int, 'tot_deaths': int, 'new_deaths': int},
                         parse_dates=['start_date', 'end_date'])

    ox_df = pd.read_csv('OxCGRT_USA_latest.csv', dtype={'RegionCode': str}, parse_dates=['Date'])[
        ['Date', 'RegionCode', 'ContainmentHealthIndex_Average']]
    ox_df['RegionCode'] = ox_df.apply(
        lambda x: x['RegionCode'].split('_')[1] if type(x['RegionCode']) == str else x['RegionCode'], axis=1)
    df = pd.merge(cov_df, ox_df, how='inner', left_on=['start_date', 'state'], right_on=['Date', 'RegionCode'])

    df = df[df['state'] == region]
    dates = df['start_date'].values
    cases = df['new_cases'].values
    # deaths = df['new_deaths'].values
    containment = df['ContainmentHealthIndex_Average'].values
    containment[np.isnan(containment)] = 0

    containment_multiplier = interp1d([0, 66, 100], [baseline, lockdown, 0], kind='linear', fill_value='extrapolate')
    A = np.array(containment_multiplier(containment)).reshape((len(dates), 1, 1))
    B = np.ones((len(dates), 1, 1)) * 2544025
    C = np.ones((len(dates), 1, 1))
    R = np.ones((len(dates), 1, 1)) * 1e5

    z = np.array([0]).reshape(1, 1)
    P = np.array([10]).reshape(1, 1)
    zs = []
    Ps = []
    z_hats = []
    P_hats = []

    for i in range(len(dates)):
        z, P = kalman_filter_predict(z, P, A[i], B[i])
        z_hats.append(z)
        P_hats.append(P)
        z, P = kalman_filter_update(cases[i], z, P, C[i], R[i])
        zs.append(z)
        Ps.append(P)

    zs = np.array(zs).reshape(-1)
    Ps = np.array(Ps).reshape(-1)
    z_hats = np.array(z_hats).reshape(-1)
    P_hats = np.array(P_hats).reshape(-1)
    if make_plot:
        plt.plot(dates, cases, label='True New Cases')
        plt.plot(dates, z_hats, label='Kalman Filter')
        # plt.fill_between(dates, z_hats - np.sqrt(P_hats), z_hats + np.sqrt(P_hats), alpha=0.5)
        plt.xticks(dates[::35])
        plt.legend()
        plt.title(abbr_to_state[region] + " Cases and Kalman Filter Predictions")
        plt.xlabel("Date")
        plt.ylabel("New Cases")
        plt.savefig(abbr_to_state[region] + "_kalman_filter.png")


if __name__=="__main__":
    run_individual_test('TX')

    # solution = solve_ivp(lambda t, y: R0[int(t)] * y, [0, len(dates)-1], [1], t_eval=np.arange(len(dates)), step_size=1)
    # S = solution.y[0]
    # plt.subplots(2, 1)
    # plt.subplot(2, 1, 1)
    # plt.plot(dates, cases, label='Cases')
    # plt.plot(dates, S, label='Model')
    # plt.xticks([])
    # plt.subplot(2, 1, 2)
    # plt.plot(dates, containment, label='Containment Index')
    # plt.xticks(dates[::35])
    # plt.show()

