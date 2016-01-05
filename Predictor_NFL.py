# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 15:43:01 2015

@author: lshulgin
"""

from Constants import *
import os
import pandas as pd
from Predictor import Predictor

INJURED_RESERVE = 'IR'
OUT = 'O'
DOUBTFUL = 'D'
QUESTIONABLE = 'Q'
PROBABLE = 'P'

INJURED_STATES = [INJURED_RESERVE, OUT, DOUBTFUL, QUESTIONABLE]

class Predictor_NFL(Predictor):
    pass


class Predictor_NFL_FanDuel_Dummy(Predictor_NFL):
    def _predict_internal(self):
        # Just use raw site output for now
        filename_data_fd = os.path.join(DIR_SALARIES, 'FanDuel NFL Output 27Dec15.csv')

        frame = pd.read_csv(filename_data_fd)
        
        # FanDuel insists on calling DST as 'D'
        frame[COL_POSITION] = map(lambda x: 'DST' if x == 'D' else x, frame['Position'])
        frame[COL_NAME] = map(lambda first, last: first + ' ' + last, frame['First Name'], frame['Last Name'])

        # Drop injured players
        frame = frame[~frame["Injury Indicator"].isin(INJURED_STATES)]
        
        column_map = {
            COL_NAME : COL_NAME,
            COL_POSITION : COL_POSITION,
            'Team' : COL_TEAM,
            'FPPG' : COL_POINTS,
            'Salary' : COL_SALARY
        }
        
        frame.rename(columns = column_map, inplace = True)
        
        return frame[column_map.values()]    


class Predictor_NFL_DraftKings_Dummy(Predictor_NFL):
    def _predict_internal(self):
        # Just use raw site output for now
        filename_data_fd = os.path.join(DIR_SALARIES, 'DraftKings NFL Output 27Dec15.csv')

        frame = pd.read_csv(filename_data_fd)

        # No injury indicator!
        
        column_map = {
            'Name' : COL_NAME,
            'Position' : COL_POSITION,
            'teamAbbrev' : COL_TEAM,
            'AvgPointsPerGame' : COL_POINTS,
            'Salary' : COL_SALARY,
        }        
        
        frame.rename(columns = column_map, inplace = True)
        
        return frame[column_map.values()]