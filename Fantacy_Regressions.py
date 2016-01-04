import nbastats.nbastats as NBA
import goldsberry
import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import datasets
from sklearn import preprocessing
pd.options.display.float_format = '${:,.2f}'.format
pd.options.mode.chained_assignment = None  # default='warn'

TEAM_DICT = {\
		'1610612743': 'DEN',\
		'1610612740': 'NOP',\
		'1610612758': 'SAC',\
		'1610612741': 'CHI',\
		'1610612737': 'ATL',\
		'1610612744': 'GSW',\
		'1610612745': 'HOU',\
		'1610612765': 'DET',\
		'1610612749': 'MIL',\
		'1610612757': 'POR',\
		'1610612764': 'WAS',\
		'1610612753': 'ORL',\
		'1610612756': 'PHX',\
		'1610612759': 'SAS',\
		'1610612760': 'OKC',\
		'1610612750': 'MIN',\
		'1610612746': 'LAC',\
		'1610612742': 'DAL',\
		'1610612752': 'NYK',\
		'1610612739': 'CLE',\
		'1610612748': 'MIA',\
		'1610612762': 'UTA',\
		'1610612755': 'PHI',\
		'1610612763': 'MEM',\
		'1610612761': 'TOR',\
		'1610612754': 'IND',\
		'1610612747': 'LAL',\
		'1610612751': 'BKN',\
		'1610612766': 'CHA',\
		'1610612738': 'BOS',}
		
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

class Player:
   'Common base class for all players'

   def __init__(self, player_id, year, gamenumber=0):
      self.id = player_id
      self.year = year
      self.log = self.player_log(gamenumber)
      if self.log.empty:
         self.ifempty = 0
      else:
         self.ifempty = 1 

   
   def player_id(self):
      return self.id

   def player_year(self):
      return self.year

   def player_ifempty(self):
      return self.ifempty

   def player_info(self):
      return pd.DataFrame(goldsberry.player.demographics(self.id))

   def Player_team_id(self):
      playersCurrent = pd.DataFrame(goldsberry.PlayerList(self.year))
      Player_team_id = playersCurrent[playersCurrent['PERSON_ID'] == self.id].TEAM_ID
      return Player_team_id

   def player_log(self, gamenumber=0):
      player_log = goldsberry.player.game_logs(self.id, season=self.year)
      player_log = pd.DataFrame(player_log.logs())
      if not player_log.empty:
         player_log = self.__AddFantacyPointToPlayerLog(player_log)

      player_log = player_log[gamenumber:]
      player_log = player_log.reset_index(drop=True)
      return player_log

   def player_gamePlayed(self):
      return len(self.log.index)

   def __AddFantacyPointToPlayerLog(self, Player_log):
      NewPlayer_log = pd.DataFrame()
      
      if not Player_log.empty:
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

   def player_log_FPTS(self):
      player_log_FPTS = pd.DataFrame()
      if self.ifempty == 1:
         player_log_FPTS = self.log[['Game_ID','FPTS']]
      return player_log_FPTS

   def Player_Forecast_Get_FPTS_Diff(self, diffNumber = 1):
      Player_Forecast_Get_FPTS_Diff = pd.DataFrame()
      if self.ifempty == 1:
         Player_Forecast_Get_FPTS_Diff = self.player_log_FPTS()
         Player_Forecast_Get_FPTS_Diff["FPTS_Diff"+str(diffNumber)] = Player_Forecast_Get_FPTS_Diff["FPTS"].diff(-diffNumber)
         Player_Forecast_Get_FPTS_Diff = Player_Forecast_Get_FPTS_Diff.drop('FPTS',1)
         Player_Forecast_Get_FPTS_Diff = Player_Forecast_Get_FPTS_Diff[np.isfinite(Player_Forecast_Get_FPTS_Diff["FPTS_Diff"+str(diffNumber)])]
      return Player_Forecast_Get_FPTS_Diff

   def Player_Forecast_Get_WinLoss(self):
      Player_Forecast_Get_WinLoss = pd.DataFrame()
      if self.ifempty == 1:
         Player_Forecast_Get_WinLoss = self.log[['Game_ID','WL']]
         Player_Forecast_Get_WinLoss = Player_Forecast_Get_WinLoss.replace(['W','L'],[1,0])
      return Player_Forecast_Get_WinLoss

   def Player_Forecast_Get_HomeAway(self):
      Player_Forecast_Get_HomeAway = pd.DataFrame()
      if self.ifempty == 1:
         Player_Forecast_Get_HomeAway = self.log[['Game_ID','MATCHUP']]
         size = len(Player_Forecast_Get_HomeAway.index)
         for x in range(0, size):
            if "vs." in Player_Forecast_Get_HomeAway['MATCHUP'][x]:
               Player_Forecast_Get_HomeAway['MATCHUP'][x] = 1
            else:
               Player_Forecast_Get_HomeAway['MATCHUP'][x] = 0
      return Player_Forecast_Get_HomeAway

   def Player_Forecast_Get_LastMinutes(self):
      Player_Forecast_Get_LastMinutes = pd.DataFrame()
      if self.ifempty == 1:
         Player_Forecast_Get_LastMinutes = self.log[['Game_ID','MIN']]
      return Player_Forecast_Get_LastMinutes

   def Player_Forecast_Get_Minutes(self):
      Player_Forecast_Get_Minutes = pd.DataFrame()
      if self.ifempty == 1:
         Player_Forecast_Get_Minutes = self.log[['Game_ID','MIN']]
      return Player_Forecast_Get_Minutes

class Team:

   def __init__(self, team_id, year, gamenumber=0):
      self.id = team_id
      self.year = year
      self.playerInfo = self.team_allPlayers_info()
      self.log = self.team_log(gamenumber)

   def team_id(self):
      return self.id

   def team_year(self):
      return self.year

   def team_allPlayers_info(self):
      team_allPlayers_info = goldsberry.team.roster(self.id, season=self.year)
      team_allPlayers_info = pd.DataFrame(team_allPlayers_info.players())
      return team_allPlayers_info

   def team_log(self, gamenumber=0):
      team_log = goldsberry.team.game_logs(self.id, season=self.year)
      team_log = pd.DataFrame(team_log.logs())
      team_log = self.__AddFantacyPointToTeamLog(team_log)

      team_log = team_log[gamenumber:]
      team_log = team_log.reset_index(drop=True)
      return team_log

   def team_gamePlayed(self):
      return len(self.log.index)

   def __AddFantacyPointToTeamLog(self, team_log):
      NewTeam_log = pd.DataFrame()
      
      if not team_log.empty:
         fantacy_point = team_log[['Game_ID', 'PTS']]
         fantacy_point.columns = ['Game_ID', 'FPTS']
         for x in range(0, len(fantacy_point.index) - 1):
            game_num = x
            fpts = CalFantacyPoint(team_log['PTS'][game_num],\
                                     team_log['FG3M'][game_num],\
                                     team_log['REB'][game_num],\
                                     team_log['AST'][game_num],\
                                     team_log['STL'][game_num],\
                                     team_log['BLK'][game_num],\
                                     team_log['TOV'][game_num])
            # an approximated way to adjust for the tripledouble will need to change
            fantacy_point['FPTS'][x] = fpts - 3
         NewTeam_log = pd.merge(team_log, fantacy_point, on='Game_ID', how='left')
      return NewTeam_log

   def team_log_FPTS(self):
      team_log_FPTS = pd.DataFrame()
      if self.ifempty == 1:
         team_log_FPTS = self.log[['Game_ID','FPTS']]
      return team_log_FPTS

   def team_Forecast_allPlayers_FPTS(self, minGames):
      team_Forecast_allPlayers_FPTS = pd.Series([])
      player_list = self.playerInfo['PLAYER_ID']
      for x in range(0, len(player_list.index)):
         player = Player(player_list[x], self.year)
         if player.player_ifempty() == 1:
            player_log_FPTS = player.player_log_FPTS()
            player_log_FPTS.columns = ['Game_ID', player.player_id()]
            if len(player_log_FPTS.index) >= minGames:
               if team_Forecast_allPlayers_FPTS.empty:
                  team_Forecast_allPlayers_FPTS = player_log_FPTS
               else:
                  team_Forecast_allPlayers_FPTS = pd.merge(team_Forecast_allPlayers_FPTS,player_log_FPTS, how='outer', on='Game_ID')
      team_Forecast_allPlayers_FPTS = team_Forecast_allPlayers_FPTS.sort('Game_ID', ascending=[False])
      team_Forecast_allPlayers_FPTS = team_Forecast_allPlayers_FPTS.reset_index(drop=True)
      return team_Forecast_allPlayers_FPTS

class Analysis:
   def __init__(self, factors_dict, player):
      self.factors_dict = factors_dict
      self.player = player
      self.prepared_results = self.analysis_prepare_basedOnFactorDict()


   def analysis_prepare_basedOnFactorDict(self):
      player = self.player
      prepared_results = player.player_log_FPTS()

      if self.factors_dict['WINLOSS'] == 1:
         temp = player.Player_Forecast_Get_WinLoss()
         prepared_results = pd.merge(prepared_results,temp, how='left', on='Game_ID')
      if self.factors_dict['HOMEAWAY'] == 1:
         temp = player.Player_Forecast_Get_HomeAway()
         prepared_results = pd.merge(prepared_results,temp, how='left', on='Game_ID')
      if self.factors_dict['LASTMINUTES'] == 1:
         temp = player.Player_Forecast_Get_LastMinutes()
         prepared_results = pd.merge(prepared_results,temp, how='left', on='Game_ID')
      return prepared_results

   def analysis_prepare_dependent_Diff(self):
      prepared_results = self.prepared_results[['Game_ID', 'FPTS_Diff1']]
      return prepared_results

   def analysis_prepare_dependent(self):
      prepared_results = self.prepared_results[['Game_ID', 'FPTS']]
      prepared_results = prepared_results[0:len(prepared_results)-1]
      prepared_results = prepared_results.reset_index(drop=True)
      return prepared_results

   def analysis_prepare_independent_All(self):
      if self.factors_dict['LASTFPTS'] == 0:
      	prepared_results = self.prepared_results.drop('FPTS', 1)
      else:
      	prepared_results = self.prepared_results
      prepared_results = prepared_results[1:]
      prepared_results = prepared_results.reset_index(drop=True)
      return prepared_results

   def analysis_prepare_independent_Last(self):
      if self.factors_dict['LASTFPTS'] == 0:
      	prepared_results = self.prepared_results.drop('FPTS', 1)
      else:
      	prepared_results = self.prepared_results
      prepared_results = prepared_results[0:1]
      prepared_results = prepared_results.reset_index(drop=True)
      return prepared_results

def regressionMethods(independent, dependent, regType=0):
	if regType == 0:
		clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
	elif regType == 1:
		clf = linear_model.LassoCV(alphas=[0.1, 1.0, 10.0])
	elif  regtype == 2:
		clf = linear_model.LassoLarsIC(criterion='bic')
	elif  regType == 3:
		clf = linear_model.ElasticNetCV(alphas=[0.1, 1.0, 10.0])
		
	clf.fit (independent, dependent)
	return clf

class Forecast:
   def __init__(self, analysis, instructions_dict=0):
      self.instructions_dict = instructions_dict
      self.analysis = analysis
      self.dependent = self.__prepare_data_dependent()
      self.independent = self.__prepare_data_independent()
      self.independent_forecast = self.__prepare_data_independentForecast()

   def __prepare_data_dependent(self):
			dependent = self.analysis.analysis_prepare_dependent()
			dependent = dependent.sort('Game_ID', ascending=[True])
			dependent = dependent.reset_index(drop=True)
			dependent = dependent.drop('Game_ID', 1)
			dependent = dependent.as_matrix()
			dependent = np.squeeze(np.asarray(dependent))
			return dependent

   def __prepare_data_independent(self):
			independent = self.analysis.analysis_prepare_independent_All()
			independent = independent.sort('Game_ID', ascending=[True])
			independent = independent.reset_index(drop=True)
			independent = independent.drop('Game_ID', 1)
			independent = independent.as_matrix()
			independent = preprocessing.scale(independent)
			return independent

   def __prepare_data_independentForecast(self):
			independent_forecast = self.analysis.analysis_prepare_independent_Last()
			independent_forecast = independent_forecast.drop('Game_ID', 1)
			independent_forecast = independent_forecast.values.tolist()
			independent_forecast = preprocessing.scale(independent_forecast)
			return independent_forecast

   def forecast_model_1(self, dependent=[], independent=[], independent_forecast=[], method = 1):
			if not dependent:
				dependent = self.dependent
			if not independent:
				independent = self.independent
			if not independent_forecast:
				independent_forecast = self.independent_forecast

      # Use Regressions to fit first
			reg_fit = regressionMethods(independent, dependent, method)
			predict = reg_fit.predict(independent)
			reg_prediction = reg_fit.predict(independent_forecast)

			# Use ARMA model to fit the resedual
			dependent = pd.DataFrame(dependent, columns = ['original'])
			predict = pd.DataFrame(predict, columns = ['predict'])
			different = dependent['original'] - predict['predict']
			different = different.as_matrix()
			arma_fit = sm.tsa.ARMA(different, (1,0)).fit()
			arma_prediction = arma_fit.predict(start=len(different), end=len(different))

			# Combine the two to get final prediction
			final_prediction = reg_prediction+arma_prediction
			return final_prediction

   def forecast_get_correlationMatrix(self, dataMatrix):
      dataMatrix = dataMatrix.drop('Game_ID', 1)
      correlationMatrix = dataMatrix.corr(dataMatrix)
      return correlationMatrix



def GetOnePlayerForecasts(player_ID = 201935):
	FACTORS_DICT = \
	{
	   'LASTFPTS': 1,\
	   'WINLOSS': 1,\
	   'HOMEAWAY': 1,\
	   'LASTMINUTES': 1
	}
	player = Player(player_ID, '2015', 0)
	analysis = Analysis(FACTORS_DICT,player)
	forecast = Forecast(analysis)

	final_prediction = forecast.forecast_model_1(method=1)
	return final_prediction

result = GetOnePlayerForecasts()
print result

def runResults():
	FACTORS_DICT = \
	{
	   'AR1': 1,\
	   'WINLOSS': 1,\
	   'HOMEAWAY': 1,\
	   'LASTMINUTES': 1
	}

	playersCurrent = pd.DataFrame(goldsberry.PlayerList(2015))
	playersCurrent.to_csv('playerList.csv')
	teamList = playersCurrent[['TEAM_CODE', 'TEAM_ID']]
	teamList = teamList.drop_duplicates(take_last=True)
	teamList['TEAM_CODE'].replace('', np.nan, inplace=True)
	teamList.dropna(subset=['TEAM_CODE'], inplace=True)
	teamList = teamList.reset_index(drop=True)

	Results = [['test', 0]]
	# for x in range(0, len(teamList.index)):
	for x in range(0, len(teamList.index)):
	   Team_id = teamList['TEAM_ID'][x]
	   Team_roster = goldsberry.team.roster(Team_id, season='2015')
	   Team_roster = pd.DataFrame(Team_roster.players())

	   for y in range(0, len(Team_roster.index)):
	   # for y in range(0, len(Team_roster.index)):   
	      Player_id = Team_roster['PLAYER_ID'][y]
	      player = Player(Player_id, '2015')
	      forecastnum = 0
	      try:
	         if player.player_gamePlayed>=20:
	         	forecast = GetOnePlayerForecasts(Player_id)
	         	forecastnum = forecast[0]
	         Results.append([Player_id,forecastnum])

	      except Exception: 
	         Results.append([Player_id,forecastnum])
	         pass

	Results = pd.DataFrame(Results, columns=['PLAYER_ID', 'ForecastFPTS'])
	Results.to_csv('result2.csv')







player_ID = 201976

FACTORS_DICT = \
	{
	   'LASTFPTS': 1,\
	   'WINLOSS': 1,\
	   'HOMEAWAY': 1,\
	   'LASTMINUTES': 1
	}
emp1 = Player(player_ID, '2015', 0)
historical = emp1.player_log_FPTS()
historical = historical.sort('Game_ID', ascending=[True])
historical = historical.reset_index(drop=True)
historical = historical.drop('Game_ID', 1)
historical['Forecast'] = 0
historical.loc[len(historical)] = [0,0]

result = [0]
for x in range(0, 1):
	emp1 = Player(player_ID, '2015', x)
	analysis = Analysis(FACTORS_DICT,emp1)
	dependent = analysis.analysis_prepare_dependent()
	independent = analysis.analysis_prepare_independent_All()
	independent_forecast = analysis.analysis_prepare_independent_Last()


	dependent = dependent.sort('Game_ID', ascending=[True])
	dependent = dependent.reset_index(drop=True)
	dependent = dependent.drop('Game_ID', 1)
	# dependent = dependent[0:len(dependent)-1]
	dependent = dependent.as_matrix()
	dependent = np.squeeze(np.asarray(dependent))

	

	independent = independent.sort('Game_ID', ascending=[True])
	independent = independent.reset_index(drop=True)
	independent = independent.drop('Game_ID', 1)
	# independent = independent.drop('FPTS_Diff1', 1)
	independent = independent.as_matrix()

	independent = preprocessing.scale(independent)

	independent_forecast = independent_forecast.drop('Game_ID', 1)
	# independent_forecast = independent_forecast.drop('FPTS_Diff1', 1)
	independent_forecast = independent_forecast.values.tolist()
	independent_forecast = preprocessing.scale(independent_forecast)


	clf = regressionMethods(independent, dependent, 1)
	predict = clf.predict(independent)



	# print dependent
	print independent
	print independent_forecast
	point_predict = clf.predict(independent_forecast)

	dependent = pd.DataFrame(dependent, columns = ['original'])
	predict = pd.DataFrame(predict, columns = ['predict'])

	different = dependent['original'] - predict['predict']
	different = different.as_matrix()
	model_fitted = sm.tsa.ARMA(different, (1,0)).fit()
	model_forecast = model_fitted.predict(start=0, end=len(different)-1)

	model_forecast_point = model_fitted.predict(start=len(different), end=len(different))

	different2 = different - model_forecast

	different = pd.DataFrame(different, columns = ['original'])
	model_forecast = pd.DataFrame(model_forecast, columns = ['predict'])
	different2 = pd.DataFrame(different2, columns = ['different'])

	final_prediction = predict['predict'] + model_forecast['predict']
	final_prediction = final_prediction.as_matrix()
	final_prediction = pd.DataFrame(final_prediction, columns = ['predict'])

	# print different

	final_prediction_point = point_predict+model_forecast_point

	dependent['original'].plot()
	final_prediction['predict'].plot()
	plt.show()

	print final_prediction_point



	# print different
	# print model_forecast
	# different['original'].plot()
	# model_forecast['predict'].plot()
	# different2['different'].plot()
	# # sm.graphics.tsa.plot_acf(different, lags=10)
	# # sm.graphics.tsa.plot_pacf(different, lags=10)
	# plt.show()


   # forecast = Forecast(FACTORS_DICT)
   # forecastnum = forecast.forecast_model_ARIMA(dependent, independent, independent_forecast)
   
   # forecastnum = forecast.forecast_model_ARIMA(dependent)
   # temp = forecastnum[0]
   # historical['Forecast'][len(historical.index)-1-x] = temp
   # result.append(temp)



# print historical
# print result

# historical['FPTS'].plot()
# historical['Forecast'].plot()
# plt.show()



