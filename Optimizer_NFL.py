# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 13:48:05 2015

@author: lshulgin
"""

from Constants import *
from Optimizer import Optimizer
import os
import pandas as pd


QB = 'QB'
RB = 'RB'
WR = 'WR'
TE = 'TE'
K = 'K'
DST = 'DST'
FLEX = 'FLEX'

POSITION_MAP = {
    FLEX : [RB, WR, TE]
}

class Optimizer_NFL(Optimizer):
    def __init__(self, frame):
        self._position_map = POSITION_MAP
        super(Optimizer_NFL, self).__init__(frame)

class Optimizer_NFL_FanDuel(Optimizer_NFL):
    def __init__(self, frame):
        self._salary_cap = 60000
        self._positions = [QB, RB, RB, WR, WR, WR, TE, K, DST]
        super(Optimizer_NFL_FanDuel, self).__init__(frame)

    # Can we replace this with an API call?    
    def _get_salaries(self):
        dir_input = os.path.join(DIR_ROOT, 'Salaries')
        filename_data_fd = os.path.join(dir_input, 'FanDuel NFL Output 27Dec15.csv')

        frame = pd.read_csv(filename_data_fd)
        
        # FanDuel insists on calling DST as 'D'
        frame[COL_POSITION] = map(lambda x: 'DST' if x == 'D' else x, frame['Position'])
        frame[COL_NAME] = map(lambda first, last: first + ' ' + last, frame['First Name'], frame['Last Name'])
        
        column_map = {
            COL_NAME : COL_NAME,
            COL_POSITION : COL_POSITION,
            'Team' : COL_TEAM,
            'Salary' : COL_SALARY
        }
        
        frame.rename(columns = column_map, inplace = True)
        
        return frame[column_map.values()]

class Optimizer_NFL_DraftKings(Optimizer_NFL):
    def __init__(self, frame):
        self._salary_cap = 50000
        self._positions = [QB, RB, RB, WR, WR, WR, TE, FLEX, DST]
        super(Optimizer_NFL_DraftKings, self).__init__(frame)
    
    
    # Can we replace this with an API call?    
    def _get_salaries(self):
        dir_input = os.path.join(DIR_ROOT, 'Salaries')
        filename_data_fd = os.path.join(dir_input, 'DraftKings NFL Output 27Dec15.csv')

        frame = pd.read_csv(filename_data_fd)
        
        column_map = {
            'Name' : COL_NAME,
            'Position' : COL_POSITION,
            'teamAbbrev' : COL_TEAM,
            'Salary' : COL_SALARY,
        }        
        
        frame.rename(columns = column_map, inplace = True)
        
        return frame[column_map.values()]