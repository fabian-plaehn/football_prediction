import pickle
import time
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import re
from selenium.webdriver.common.by import By
from club_list import club_list

driver = webdriver.Edge()
df = None
until_year = 2020

for club_data in club_list:
    link = "https://int.soccerway.com/teams/" + club_data.country + "/" + club_data.club_name + "/" + club_data.number + "/"

    wins = []
    losses = []
    draws = []
    dates = []

    driver.get(link)
    time.sleep(3)

    for i in range(club_data.max_try_cookie):  # more option css-sob0ma # false css-1dcy74u # accept css-1dcy74u
        try:
            driver.find_element(by=By.CLASS_NAME, value=club_data.agree_id).click()
            time.sleep(0.1)
            break
        except:
            time.sleep(0.5)
            print("try again - cookie")

    time.sleep(1)

    html = driver.page_source
    soup = BeautifulSoup(html)

    while True:
        i = 0
        for i, line in enumerate(soup.find_all('td')):
            if "result-win" in str(line):
                wins.append(1)
                losses.append(0)
                draws.append(0)
            elif "result-loss" in str(line):
                wins.append(0)
                losses.append(1)
                draws.append(0)
            elif "result-draw" in str(line):
                wins.append(0)
                losses.append(0)
                draws.append(1)
            else:
                continue
            print(line)
            start_idx = [m.start() + len('/matches/') for m in re.finditer('/matches/', str(line))]
            date = str(line)[start_idx[0]:start_idx[0] + len("2023/00/00")]
            print(int(date[:4]))
            if int(date[:4]) == until_year:
                break
            dates.append(date)
        if i != len(soup.find_all('td')) - 1:
            wins.pop(-1)
            losses.pop(-1)
            draws.pop(-1)
            break
        while True:
            try:
                python_button = driver.find_element(value="page_team_1_block_team_matches_summary_11_previous")
                python_button.click()  # click load more button
                time.sleep(1)
                if html == driver.page_source:
                    raise Exception
                soup = BeautifulSoup(driver.page_source)
                break
            except:
                time.sleep(0.5)
                print("try again")
        print("success")

    df_temp = pd.DataFrame(list(zip(wins, losses, draws)), index=pd.to_datetime(dates, format="%Y/%m/%d"), columns=[f"{club_data.club_name}_win", f"{club_data.club_name}_loss", f"{club_data.club_name}_draw"])
    df_temp = df_temp[~df_temp.index.duplicated(keep="first")]  # let's hope we don't need this
    if df is None:
        df = df_temp
    else:
        df = pd.concat([df, df_temp], axis=1)
    print(df.shape)

df.fillna(0, inplace=True)
print(df)
with open("team_data.pickle", "wb") as f:
    pickle.dump(df, f)
