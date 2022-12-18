import matplotlib.pyplot as plt
import pandas as pd
import datetime


def plot_state_cases(state_abbrs):
    df = pd.read_csv('Weekly_United_States_COVID-19_Cases_and_Deaths_by_State (1).csv',
                     dtype={'State': str, 'start_date': str, 'end_date': str,
                            'tot_cases': int, 'new_cases': int, 'tot_deaths': int, 'new_deaths': int},
                     parse_dates=['start_date', 'end_date'])
    plt.subplots(2, 1)
    for state in state_abbrs:
        indices = df['state'] == state
        plt.subplot(2, 1, 1)
        plt.plot(df[indices]['start_date'], df[indices]['new_cases'], label=state)
        plt.xticks([])
        plt.title("New Cases")
        plt.ylabel("Number of Cases per Week")
        plt.yticks([0, 2e5, 4e5, 6e5, 8e5], labels=['0', '200k', '400k', '600k', '800k'])
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(df[indices]['start_date'], df[indices]['new_deaths'], label=state)
        plt.xticks(df[indices]['start_date'][::35])
        plt.title("New Deaths")
        plt.xlabel("Date")
        plt.ylabel("Number of Deaths per Week")
        plt.legend()
    # plt.savefig('Figure2.png')
    plt.show()


def plot_state_meausres(state_abbrs):
    df = pd.read_csv('OxCGRT_USA_latest.csv', parse_dates=['Date'])
    for state in state_abbrs:
        indices = df['RegionCode'] == state
        plt.plot(df[indices]['Date'], df[indices]['ContainmentHealthIndex_Average'], label='C1')
        plt.title("Containment Index")
        plt.xlabel("Date")
        plt.ylabel("Containment Index")
        plt.xticks(df[indices]['Date'][::200])
        plt.legend()
        plt.show()


if __name__=="__main__":
    # plot_state_meausres(['US_CA'])
    plot_state_cases(['CA'])