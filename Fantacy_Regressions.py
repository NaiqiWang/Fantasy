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

   def __init__(self, team_id, year):
      self.id = team_id
      self.year = year
      self.playerInfo = self.team_allPlayers_info()

   def team_id(self):
      return self.id

   def team_year(self):
      return self.year

   def team_allPlayers_info(self):
      team_allPlayers_info = goldsberry.team.roster(self.id, season=self.year)
      team_allPlayers_info=pd.DataFrame(team_allPlayers_info.players())
      return team_allPlayers_info
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
      # if self.factors_dict['AR1'] == 1:
      #    temp = player.Player_Forecast_Get_FPTS_Diff()
      #    prepared_results = pd.merge(prepared_results,temp, how='left', on='Game_ID')
      #    prepared_results = prepared_results[np.isfinite(prepared_results['FPTS_Diff1'])]
      #    prepared_results = prepared_results.reset_index(drop=True)
      # if self.factors_dict['LASTFPTS'] == 1:
      #    temp = player.Player_Forecast_Get_WinLoss()
      #    prepared_results = pd.merge(prepared_results,temp, how='left', on='Game_ID')
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
      prepared_results = prepared_results[1:]
      prepared_results = prepared_results.reset_index(drop=True)
      return prepared_results

   def analysis_prepare_independent_Last(self):
      if self.factors_dict['LASTFPTS'] == 0:
      	prepared_results = self.prepared_results.drop('FPTS', 1)
      prepared_results = prepared_results[0:1]
      prepared_results = prepared_results.reset_index(drop=True)
      return prepared_results

class Forecast:
   def __init__(self, instructions_dict=0):
      self.instructions_dict = instructions_dict

   def forecast_model_ARIMA(self, dependent, independent=pd.Series([]), independent_forecast=pd.Series([])):
      dependent = dependent.sort('Game_ID', ascending=[True])
      dependent = dependent.reset_index(drop=True)
      dependent = dependent.drop('Game_ID', 1)
      dependent = dependent[0:len(dependent)-1]
      dependent = list(dependent.values)

      if not independent.empty:
         independent = independent.sort('Game_ID', ascending=[True])
         independent = independent.reset_index(drop=True)
         independent = independent.drop('Game_ID', 1)
         independent = independent.drop('FPTS_Diff1', 1)
         independent = independent.values.tolist()

         independent_forecast = independent_forecast.drop('Game_ID', 1)
         independent_forecast = independent_forecast.drop('FPTS_Diff1', 1)
         independent_forecast = independent_forecast.values.tolist()

         model_fitted = sm.tsa.ARIMA(dependent, (5,0,0), exog=independent).fit()

         model_forecast = model_fitted.predict(start=len(dependent), end=len(dependent), exog=independent_forecast)
      else:
         model_fitted = sm.tsa.ARIMA(dependent, (1,0,0)).fit()
         model_forecast = model_fitted.predict(start=len(dependent), end=len(dependent))
      #return model_fitted
      # model_forecast = model_fitted.predict(start=len(dependent), end=len(dependent))
      return model_forecast

   def forecast_get_correlationMatrix(self, dataMatrix):
      dataMatrix = dataMatrix.drop('Game_ID', 1)
      correlationMatrix = dataMatrix.corr(dataMatrix)
      return correlationMatrix


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


FACTORS_DICT = \
{
   'LASTFPTS': 0,\
   'WINLOSS': 1,\
   'HOMEAWAY': 1,\
   'LASTMINUTES': 1
}
emp1 = Player(201935, '2015', 0)
historical = emp1.player_log_FPTS()
historical = historical.sort('Game_ID', ascending=[True])
historical = historical.reset_index(drop=True)
historical = historical.drop('Game_ID', 1)
historical['Forecast'] = 0
historical.loc[len(historical)] = [0,0]

result = [0]
for x in range(0, 1):
	emp1 = Player(201935, '2015', x)
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

	clf = regressionMethods(independent, dependent, 1)
	predict = clf.predict(independent)



	print dependent
	print independent
	# clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
	# clf.fit (independent, dependent)
	# predict = clf.predict(independent)

	# clf = linear_model.LassoCV(alphas=[0.1, 1.0, 10.0])
	# clf.fit (independent, dependent)
	# predict = clf.predict(independent)


	# clf = linear_model.LassoLarsIC(criterion='bic')
	# clf.fit (independent, dependent)
	# predict = clf.predict(independent)


	dependent = pd.DataFrame(dependent, columns = ['original'])
	predict = pd.DataFrame(predict, columns = ['predict'])

	# print clf.alpha_
	# print clf.coef_

	# dependent['original'].plot()
	# predict['predict'].plot()
	# plt.show()

	different = dependent['original'] - predict['predict']
	different = different.as_matrix()
	model_fitted = sm.tsa.ARMA(different, (1,0)).fit()
	model_forecast = model_fitted.predict(start=0, end=len(different)-1)


	different2 = different - model_forecast

	different = pd.DataFrame(different, columns = ['original'])
	model_forecast = pd.DataFrame(model_forecast, columns = ['predict'])
	different2 = pd.DataFrame(different2, columns = ['different'])

	final_prediction = predict['predict'] + model_forecast['predict']
	final_prediction = final_prediction.as_matrix()
	final_prediction = pd.DataFrame(final_prediction, columns = ['predict'])

	print different

	dependent['original'].plot()
	final_prediction['predict'].plot()
	plt.show()


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

























clf = linear_model.Ridge (alpha = .1)
# clf.set_params(alpha=0.5, copy_X=True, fit_intercept=True, max_iter=None,
#       normalize=False, random_state=None, solver='auto', tol=0.001)
clf.fit ([[0, 1], [0, 0], [1, 1]], [0, .1, 1])

print clf.coef_


clf = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
clf.fit([[0, 1, 1, 3], [0, 0, 3, 5], [1, 1, 1, 0], [1, 1, 1, 3]], [0, .1, 1, 3])

print clf.coef_
print clf.alpha_










# print(__doc__)

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import linear_model

# # X is the 10x10 Hilbert matrix
# X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
# y = np.ones(10)

# ###############################################################################
# # Compute paths

# n_alphas = 200
# alphas = np.logspace(-10, -2, n_alphas)
# clf = linear_model.Ridge(fit_intercept=False)

# coefs = []
# for a in alphas:
#     clf.set_params(alpha=a)
#     clf.fit(X, y)
#     coefs.append(clf.coef_)

# ###############################################################################
# # Display results

# ax = plt.gca()
# ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])

# ax.plot(alphas, coefs)
# ax.set_xscale('log')
# ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
# plt.xlabel('alpha')
# plt.ylabel('weights')
# plt.title('Ridge coefficients as a function of the regularization')
# plt.axis('tight')
# plt.show()