import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import geopandas as gpd
import time

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
    return list(state['STATE_NAME'].values)


def run_group_test(regions, test, baseline, lockdown, r, b, df, main_region=None):
    df_regions = [df[df['state'] == region] for region in regions]
    dates = df_regions[0]['start_date'].values
    cases = np.array([df_region['new_cases'].values for df_region in df_regions])
    containments = np.array([df_region['ContainmentHealthIndex_Average'].values for df_region in df_regions])
    containments[np.isnan(containments)] = 0

    containment_multiplier = interp1d([0, 66, 100], [baseline, lockdown, 0], kind='linear', fill_value='extrapolate')
    R0 = containment_multiplier(containments)
    state_list = get_state_list()
    travel_indices = [state_list.index(abbr_to_state[region]) for region in regions]
    full_travel_matrix = np.loadtxt('travel.csv', delimiter=',', skiprows=0)
    travel_matrix = full_travel_matrix[travel_indices, :][:, travel_indices]
    travel_matrix_normalized = travel_matrix / np.sum(travel_matrix, axis=1)
    A = np.repeat(R0, len(regions), axis=0).reshape((len(regions), len(regions), len(dates))) * travel_matrix_normalized.reshape((len(regions), len(regions), 1))
    B = np.ones((len(dates), len(regions), len(regions))) * b * travel_matrix_normalized
    C = np.eye(len(regions))
    R = np.eye(len(regions)) * r
    A = A.transpose((2, 0, 1))

    z = np.zeros((len(regions), 1))
    P = np.eye(len(regions)) * 10
    zs = []
    Ps = []
    z_hats = []
    P_hats = []

    for i in range(len(dates)):
        z, P = kalman_filter_predict(z, P, A[i], B[i])
        z_hats.append(z)
        P_hats.append(P)
        z, P = kalman_filter_update(cases[:, i].reshape((len(regions), 1)), z, P, C, R)
        zs.append(z)
        Ps.append(P)


    # zs = np.array(zs).reshape((len(dates), len(regions)))
    # Ps = np.array(Ps).reshape((len(dates), len(regions), len(regions)))
    z_hats = np.array(z_hats).reshape((len(dates), len(regions)))
    # P_hats = np.array(P_hats).reshape((len(dates), len(regions), len(regions)))

    if main_region is not None:
        main_region_index = regions.index(main_region)
        plt.figure()
        plt.plot(dates, cases[main_region_index], label='True New Cases')
        plt.plot(dates, z_hats[:, main_region_index], label='Kalman Filter')
        plt.xticks(dates[::35])
        plt.legend()
        plt.title(abbr_to_state[main_region] + ' ' + test)
        plt.xlabel("Date")
        plt.ylabel("New Cases")
        plt.savefig(abbr_to_state[main_region] + ' ' + test + ".png")

    return z_hats, cases


def run_individual_test(region, test, baseline, lockdown, r, b, df, make_plot=False):
    df_region = df[df['state'] == region]
    dates = df_region['start_date'].values
    cases = df_region['new_cases'].values
    # deaths = df_region['new_deaths'].values
    containment = df_region['ContainmentHealthIndex_Average'].values
    containment[np.isnan(containment)] = 0

    containment_multiplier = interp1d([0, 66, 100], [baseline, lockdown, 0], kind='linear', fill_value='extrapolate')
    A = np.array(containment_multiplier(containment)).reshape((len(dates), 1, 1))
    B = np.ones((len(dates), 1, 1)) * b #2544025
    C = np.ones((len(dates), 1, 1))
    R = np.ones((len(dates), 1, 1)) * r #1e5

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
        plt.figure()
        plt.plot(dates, cases, label='True New Cases')
        plt.plot(dates, z_hats, label='Kalman Filter')
        # plt.fill_between(dates, z_hats - np.sqrt(P_hats), z_hats + np.sqrt(P_hats), alpha=0.5)
        plt.xticks(dates[::35])
        plt.legend()
        plt.title(abbr_to_state[region] + ' ' + test)
        plt.xlabel("Date")
        plt.ylabel("New Cases")
        plt.savefig(abbr_to_state[region] + ' ' + test + ".png")

    return z_hats, cases


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets.T) ** 2).mean())


if __name__=="__main__":
    cov_df = pd.read_csv('Weekly_United_States_COVID-19_Cases_and_Deaths_by_State (1).csv',
                         dtype={'State': str, 'start_date': str, 'end_date': str,
                                'tot_cases': int, 'new_cases': int, 'tot_deaths': int, 'new_deaths': int},
                         parse_dates=['start_date', 'end_date'])

    ox_df = pd.read_csv('OxCGRT_USA_latest.csv', dtype={'RegionCode': str}, parse_dates=['Date'])[
        ['Date', 'RegionCode', 'ContainmentHealthIndex_Average']]
    ox_df['RegionCode'] = ox_df.apply(
        lambda x: x['RegionCode'].split('_')[1] if type(x['RegionCode']) == str else x['RegionCode'], axis=1)
    df = pd.merge(cov_df, ox_df, how='inner', left_on=['start_date', 'state'], right_on=['Date', 'RegionCode'])

    baseline = 2.69
    masking = 2.69 * .81
    lockdown = .7
    r = 1e5
    b = 2544025
    # print("texas, lit values, test")
    # start = time.time()
    # z_hats, true_z = run_individual_test('TX', 'literature values test', baseline, lockdown, r, b, df, make_plot=True)
    # print(time.time() - start, "seconds")
    # print("RMSE:", rmse(true_z, z_hats), "\n")
    #
    # print("new york, lit values, test")
    # start = time.time()
    # z_hats, true_z = run_individual_test('NY', 'literature values test', baseline, lockdown, r, b, df, make_plot=True)
    # print(time.time() - start, "seconds")
    # print("RMSE:", rmse(true_z, z_hats), "\n")
    #
    # print("texas, tuned values, test")
    # def error_for_tuning(tuning_params):
    #     baseline, lockdown, r, b = tuning_params
    #     z_hats, true_z = run_individual_test('NY', 'tuning test', baseline, lockdown, r, b, df)
    #     return rmse(true_z, z_hats)
    # start = time.time()
    # sol = minimize(error_for_tuning, [baseline, lockdown, r, b])
    # print("solution:", sol.x)
    # z_hats, true_z = run_individual_test('TX', 'tuning test', *sol.x, df, make_plot=True)
    # print(time.time() - start, "seconds")
    # print("RMSE:", rmse(true_z, z_hats), "\n")
    #
    # print("new york, tuned values, test")
    # def error_for_tuning(tuning_params):
    #     baseline, lockdown, r, b = tuning_params
    #     z_hats, true_z = run_individual_test('TX', 'individual tuning test', baseline, lockdown, r, b, df)
    #     return rmse(true_z, z_hats)
    # start = time.time()
    # sol = minimize(error_for_tuning, [baseline, lockdown, r, b])
    # print("solution:", sol.x)
    # z_hats, true_z = run_individual_test('NY', 'individual tuning test', *sol.x, df, make_plot=True)
    # print(time.time() - start, "seconds")
    # print("RMSE:", rmse(true_z, z_hats), "\n")

    print("texas with 4 other states, lit values, test")
    start = time.time()
    z_hats, true_z = run_group_test(['TX', 'NM', 'OK', 'AR', 'LA'], '5 group lit vals test', baseline, lockdown, r, b, df, main_region='TX')
    print(time.time() - start, "seconds")
    print("RMSE:", rmse(true_z[0], z_hats[...,0]), "\n")

    print("new york with 4 other states, lit values, test")
    start = time.time()
    z_hats, true_z = run_group_test(['NY', 'NJ', 'CT', 'PA', 'MA'], '5 group test', baseline, lockdown, r, b, df, main_region='NY')
    print(time.time() - start, "seconds")
    print("RMSE:", rmse(true_z[0], z_hats[...,0]), "\n")

    print("texas with 4 other states, tuned values, test")
    def error_for_tuning(tuning_params):
        baseline, lockdown, r, b = tuning_params
        z_hats, true_z = run_group_test(['NY', 'NJ', 'CT', 'PA', 'MA'], '5 group test', baseline, lockdown, r, b, df)
        return rmse(true_z, z_hats)
    start = time.time()
    sol = minimize(error_for_tuning, [baseline, lockdown, r, b])
    print("solution:", sol.x)
    z_hats, true_z = run_group_test(['TX', 'NM', 'OK', 'AR', 'LA'], '5 group test', *sol.x, df, main_region='TX')
    print(time.time() - start, "seconds")
    print("RMSE:", rmse(true_z[0], z_hats[...,0]), "\n")

    print("new york with 4 other states, tuned values, test")
    def error_for_tuning(tuning_params):
        baseline, lockdown, r, b = tuning_params
        z_hats, true_z = run_group_test(['TX', 'NM', 'OK', 'AR', 'LA'], '5 group test', baseline, lockdown, r, b, df)
        return rmse(true_z, z_hats)
    start = time.time()
    sol = minimize(error_for_tuning, [baseline, lockdown, r, b])
    print("solution:", sol.x)
    z_hats, true_z = run_group_test(['NY', 'NJ', 'CT', 'PA', 'MA'], '5 group test', *sol.x, df, main_region='NY')
    print(time.time() - start, "seconds")
    print("RMSE:", rmse(true_z[0], z_hats[...,0]), "\n")











