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

driver = webdriver.Edge()

link = "https://int.soccerway.com/teams/germany/fc-bayern-munchen/961/"

score = []
dates = []


driver.get(link)
html = driver.page_source
soup = BeautifulSoup(html)

for i in range(50):
    for line in soup.find_all('td')[1:]:
        if "result-win" in str(line):
            score.append(1)

            start_idx = [m.start() + len('/matches/') for m in re.finditer('/matches/', str(line))]
            date = str(line)[start_idx[0]:start_idx[0] + len("2023/00/00")]
            dates.append(date)
        elif "result-loss" in str(line):
            score.append(-1)

            start_idx = [m.start() + len('/matches/') for m in re.finditer('/matches/', str(line))]
            date = str(line)[start_idx[0]:start_idx[0]+len("2023/00/00")]
            dates.append(date)
        elif "result-draw" in str(line):
            score.append(0)

            start_idx = [m.start() + len('/matches/') for m in re.finditer('/matches/', str(line))]
            date = str(line)[start_idx[0]:start_idx[0]+len("2023/00/00")]
            dates.append(date)

    while True:
        try:
            python_button = driver.find_element(value="page_team_1_block_team_matches_summary_11_previous")
            python_button.click()  # click load more button
            soup = BeautifulSoup(driver.page_source)
            time.sleep(0.1)
            break
        except:
            time.sleep(0.5)
            print("try again")
    print("success")
    print(score, dates)
