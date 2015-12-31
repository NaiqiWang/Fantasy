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
      if self.factors_dict['AR1'] == 1:
         temp = player.Player_Forecast_Get_FPTS_Diff()
         prepared_results = pd.merge(prepared_results,temp, how='left', on='Game_ID')
         prepared_results = prepared_results[np.isfinite(prepared_results['FPTS_Diff1'])]
         prepared_results = prepared_results.reset_index(drop=True)
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
      return prepared_results

   def analysis_prepare_independent_All(self):
      prepared_results = self.prepared_results.drop('FPTS', 1)
      prepared_results = prepared_results[1:]
      prepared_results = prepared_results.reset_index(drop=True)
      return prepared_results

   def analysis_prepare_independent_Last(self):
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

# emp2 = Team(1610612743, '2015')
# print emp2.team_allPlayers_info()
# print emp2.team_Forecast_allPlayers_FPTS(10)



# def testRun(team_id = 1610612743, season = '2015'):
#    team = Team(team_id)



# FACTORS_DICT = \
# {
#    'AR1': 1,\
#    'WINLOSS': 1,\
#    'HOMEAWAY': 1,\
#    'LASTMINUTES': 1
# }

# emp1 = Player(201935, '2015')
# analysis = Analysis(FACTORS_DICT,emp1)

# dependent = analysis.analysis_prepare_dependent()
# independent = analysis.analysis_prepare_independent_All()
# independent_forecast = analysis.analysis_prepare_independent_Last()



# forecast = Forecast(FACTORS_DICT)

# forecastnum = forecast.forecast_model_ARIMA(dependent, independent, independent_forecast)


#          dependent['FPTS'].plot()

#          plt.show()




FACTORS_DICT = \
{
   'AR1': 1,\
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
for x in range(0, 15):
   emp1 = Player(201935, '2015', x)
   analysis = Analysis(FACTORS_DICT,emp1)
   dependent = analysis.analysis_prepare_dependent()
   independent = analysis.analysis_prepare_independent_All()
   independent_forecast = analysis.analysis_prepare_independent_Last()

   forecast = Forecast(FACTORS_DICT)
   forecastnum = forecast.forecast_model_ARIMA(dependent, independent, independent_forecast)
   # forecastnum = forecast.forecast_model_ARIMA(dependent)
   temp = forecastnum[0]
   historical['Forecast'][len(historical.index)-1-x] = temp
   result.append(temp)
print historical
print result

historical['FPTS'].plot()
historical['Forecast'].plot()
plt.show()














# FACTORS_DICT = \
# {
#    'AR1': 1,\
#    'WINLOSS': 1,\
#    'HOMEAWAY': 1,\
#    'LASTMINUTES': 1
# }

# playersCurrent = pd.DataFrame(goldsberry.PlayerList(2015))
# playersCurrent.to_csv('playerList.csv')
# teamList = playersCurrent[['TEAM_CODE', 'TEAM_ID']]
# teamList = teamList.drop_duplicates(take_last=True)
# teamList['TEAM_CODE'].replace('', np.nan, inplace=True)
# teamList.dropna(subset=['TEAM_CODE'], inplace=True)
# teamList = teamList.reset_index(drop=True)

# Results = [['test', 0]]
# # for x in range(0, len(teamList.index)):
# for x in range(0, 1):
#    Team_id = teamList['TEAM_ID'][x]
#    Team_roster = goldsberry.team.roster(Team_id, season='2015')
#    Team_roster = pd.DataFrame(Team_roster.players())

#    for y in range(0, 1):
#    # for y in range(0, len(Team_roster.index)):   
#       Player_id = Team_roster['PLAYER_ID'][y]
#       player = Player(Player_id, '2015')
#       forecastnum = 0
#       try:
#          if player.player_gamePlayed>=20:
#             analysis = Analysis(FACTORS_DICT,player)
#             dependent = analysis.analysis_prepare_dependent()
#             independent = analysis.analysis_prepare_independent_All()
#             independent_forecast = analysis.analysis_prepare_independent_Last()

#             forecast = Forecast(FACTORS_DICT)
#             forecastnum = forecast.forecast_model_ARIMA(dependent)
#             # forecastnum = forecast.forecast_model_ARIMA(dependent, independent, independent_forecast)
#             forecastnum = forecastnum[0]
#             print "Find me"
#             print forecastnum
#          Results.append([Player_id,forecastnum])

#       except Exception: 
#          Results.append([Player_id,forecastnum])
#          pass

# Results = pd.DataFrame(Results, columns=['PLAYER_ID', 'ForecastFPTS'])
# Results.to_csv('result2.csv')









# REG_INSTRUCTIONS_DICT = \
# {
#    'AR1': 1,\
#    'WINLOSS': 1,\
#    'HOMEAWAY': 1
# }
# reg = Forecast(REG_INSTRUCTIONS_DICT)

# data = emp1.Player_Forecast_Get_FPTS_Diff(1)
# print data
# sm.graphics.tsa.plot_acf(data.FPTS_Diff1, lags=10)
# plt.show()



# emp1 = Player(2749, '2015')
# data = emp1.Player_Forecast_Get_Minutes()
# print data

# data = data.sort('Game_ID', ascending=[True])
# data = data.reset_index(drop=True)
# print data

# data = list(data['MIN'].values)
# print data

# ar_fitted = sm.tsa.AR(data).fit(maxlag=1, method='mle', disp=-1)
# print ar_fitted.params
# ts_forecast = ar_fitted.predict(start=0, end=30)
# print ts_forecast


# ar_fitted = sm.tsa.AR([2,3,2,4,1,5,6]).fit(maxlag=1, method='mle', disp=-1)
# exog_data = np.column_stack([[2,3,2,4,1,5,1],[9,1,1,2,1,5,5]])
# ar_fitted2 = sm.tsa.ARIMA([2,3,2,4,1,5,6], (1,0,0), exog=exog_data).fit()


# print "ddddd"
# # print sm.tsa.AR([2,3,2,4,1,5,6]).select_order(maxlag=3, ic='aic', method='mle')


# print ar_fitted.params
# print ar_fitted2.params



# print len([2,3,2,4,1,5,6])
# print ts_forecast


# sm.graphics.tsa.plot_acf(data.MIN, lags=10)
# plt.show()

# preds=ar_res.predict(100,400,dynamic = True)


# print emp1.player_log_FPTS()
# print emp1.player_gamePlayed()
# print emp1.Player_Forecast_Get_FPTS_Diff()
# print emp1.Player_Forecast_Get_WinLoss()
# print emp1.Player_Forecast_Get_HomeAway()
# print emp1.Player_team_id()