import dataclasses
import pickle
import time
from bs4 import BeautifulSoup
from selenium import webdriver
import pandas as pd
import re
from pathlib import Path
import configparser
import datetime
import lxml
from selenium.webdriver.common.by import By

driver = webdriver.Edge()


@dataclasses.dataclass
class soccerway_club_data:
    country: str
    club_name: str
    number: str
    agree_id: str


club_list = [
             soccerway_club_data("germany", "fc-bayern-munchen", "961", "css-1dcy74u"),
             soccerway_club_data("england", "arsenal-fc", "660", "css-1v7z3z3"),
             ]
# TODO add more teams

df = None

for club_data in club_list:
    link = "https://int.soccerway.com/teams/" + club_data.country + "/" + club_data.club_name + "/" + club_data.number + "/"

    wins = []
    losses = []
    draws = []
    dates = []

    driver.get(link)
    time.sleep(5)

    for i in range(10):  # more option css-sob0ma # false css-1dcy74u # accept css-1dcy74u
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

    for i in range(3):
        for line in soup.find_all('td')[1:]:
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
            dates.append(date)

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

df.fillna(0, inplace=True)
print(df)
