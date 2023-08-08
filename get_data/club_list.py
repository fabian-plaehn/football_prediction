import dataclasses


@dataclasses.dataclass
class soccerway_club_data:
    country: str
    club_name: str
    number: str
    agree_id: str
    range_length: int
    max_try_cookie: int


europe_league_teams = [
    soccerway_club_data("germany", "fc-bayern-munchen", "961", "css-1dcy74u", range_length=5, max_try_cookie=10),  # ~ 50 for 3 years
    soccerway_club_data("england", "arsenal-fc", "660", "css-1v7z3z3", range_length=10, max_try_cookie=10),
    soccerway_club_data("netherlands", "afc-ajax", "1515", "css-1v7z3z3", range_length=10, max_try_cookie=10),
    soccerway_club_data("croatia", "nk-dinamo-zagreb", "479", "css-1v7z3z3", range_length=10, max_try_cookie=10),
    soccerway_club_data("spain", "real-madrid-club-de-futbol", "2016", "css-1v7z3z3", range_length=10, max_try_cookie=10),
    soccerway_club_data("scotland", "celtic-fc", "1898", "css-1v7z3z3", range_length=10, max_try_cookie=10),
    soccerway_club_data("portugal", "benfica", "1679", "css-1v7z3z3", range_length=10, max_try_cookie=10),
    soccerway_club_data("italy", "juventus-fc", "1242", "css-1v7z3z3", range_length=10, max_try_cookie=10),
]

top_ten_fifa = [
    soccerway_club_data("france", "stade-brestois-29", "922", "css-1dcy74u", range_length=5, max_try_cookie=10),  # ~ 50 for 3 years
    soccerway_club_data("france", "olympique-lyonnais", "884", "css-1v7z3z3", range_length=10, max_try_cookie=10),
    soccerway_club_data("france", "racing-club-de-strasbourg", "898", "css-1v7z3z3", range_length=10, max_try_cookie=10),
    soccerway_club_data("england", "fulham-football-club", "667", "css-1v7z3z3", range_length=10, max_try_cookie=10),
    soccerway_club_data("england", "manchester-united-fc", "662", "css-1v7z3z3", range_length=10, max_try_cookie=10),
    soccerway_club_data("belgium", "koninklijke-as-eupen", "245", "css-1v7z3z3", range_length=10, max_try_cookie=10),
    soccerway_club_data("belgium", "standard-de-liege", "230", "css-1v7z3z3", range_length=10, max_try_cookie=10),
    soccerway_club_data("spain", "real-club-celta-de-vigo", "2033", "css-1v7z3z3", range_length=10, max_try_cookie=10),
    soccerway_club_data("spain", "club-atletico-osasuna", "2022", "css-1v7z3z3", range_length=10, max_try_cookie=10),
    soccerway_club_data("germany", "1-fc-union-berlin", "1026", "css-1v7z3z3", range_length=10, max_try_cookie=10),
]

club_list = europe_league_teams + top_ten_fifa

