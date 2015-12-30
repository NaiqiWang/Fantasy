import nbastats.nbastats as NBA
import goldsberry
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.float_format = '${:,.2f}'.format
pd.options.mode.chained_assignment = None  # default='warn'

def CalFantacyPoint(pts, fg3m, reb, ast, stl, blk, tov):
 check = 0
 dd = 0
 td = 0
 if pts >= 10:
  check = check + 1
 if reb >= 10:
  check = check + 1
 if ast >= 10:
  check = check + 1
 if stl >= 10:
  check = check + 1
 if blk >= 10:
  check = check + 1
 if check == 2:
  dd = 1
 if check >= 3:
  td = 1

 fantacy_Point = pts + \
 fg3m * 0.5 + \
 reb * 1.25 + \
 ast * 1.5 + \
 stl * 2.0 + \
 blk * 2.0 + \
 tov * (-0.5) + \
 dd * 1.5 + \
 td * 3
 return fantacy_Point
def AddFantacyPointToPlayerLog(Player_log):
 fantacy_point = Player_log[['Game_ID', 'PTS']]
 fantacy_point.columns = ['Game_ID', 'FPTS']
 for x in range(0, len(fantacy_point.index) - 1):
  game_num = x
  fpts = CalFantacyPoint(Player_log['PTS'][game_num],\
                         Player_log['FG3M'][game_num],\
                         Player_log['REB'][game_num],\
                         Player_log['AST'][game_num],\
                         Player_log['STL'][game_num],\
                         Player_log['BLK'][game_num],\
                         Player_log['TOV'][game_num])
  fantacy_point['FPTS'][x] = fpts  
 NewPlayer_log = pd.merge(Player_log, fantacy_point, on='Game_ID', how='left')
 return NewPlayer_log



def GetAllPlayersLogData(season='2015'):
	playersCurrent = pd.DataFrame(goldsberry.PlayerList(season))
	# playersCurrent.to_csv('playerList.csv')
	teamList = playersCurrent[['TEAM_CODE', 'TEAM_ID']]
	teamList = teamList.drop_duplicates(take_last=True)
	teamList['TEAM_CODE'].replace('', np.nan, inplace=True)
	teamList.dropna(subset=['TEAM_CODE'], inplace=True)
	teamList = teamList.reset_index(drop=True)

	Results = pd.DataFrame()
	for x in range(0, len(teamList.index)):
		Team_id = teamList['TEAM_ID'][x]
		Team_roster = goldsberry.team.roster(Team_id, season)
		Team_roster = pd.DataFrame(Team_roster.players())

		for y in range(0, len(Team_roster.index)):
			Player_id = Team_roster['PLAYER_ID'][y]
			Player = goldsberry.player.game_logs(Player_id, season)
			Player_log = pd.DataFrame(Player.logs())
			if len(Player_log) >= 1:
				Player_log = AddFantacyPointToPlayerLog(Player_log)
				Results = Results.append(Player_log)	
	return Results




results = GetAllPlayersLogData()

results.to_csv('All_Players_Log.csv')