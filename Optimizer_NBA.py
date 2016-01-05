# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 13:48:05 2015

@author: lshulgin
"""

from Constants import *
from Optimizer import Optimizer
import os
import pandas as pd


C = 'C'
SG = 'SG'
PG = 'PG'
SF = 'SF'
PF = 'PF'
G = 'G'
F = 'F'
UTIL = 'UTIL'

POSITION_MAP = {
    G : [SG, PG],
    F : [SF, PF],
    UTIL : [C, SG, PG, SF, PF, G, F]
}

class Optimizer_NBA(Optimizer):
    def __init__(self, frame):
        self._position_map = POSITION_MAP
        super(Optimizer_NBA, self).__init__(frame)

class Optimizer_NBA_FanDuel(Optimizer_NBA):
    def __init__(self, frame):
        self._salary_cap = 60000
        self._positions = [C, SG, SG, PG, PG, SF, SF, PF, PF]
        super(Optimizer_NBA_FanDuel, self).__init__(frame)

    # Can we replace this with an API call?
    def _get_salaries(self):
        filename = os.path.join(DIR_SALARIES, 'FanDuel NBA Output 25Dec15.csv')
        
        frame = pd.read_csv(filename)
        frame[COL_NAME] = map(lambda first, last: "{0} {1}".format(first, last), frame['First Name'], frame['Last Name'])        
        
        column_map = {
            COL_NAME : COL_NAME,
            'Position' : COL_POSITION,
            'Team' : COL_TEAM,
            'Salary' : COL_SALARY
        }
        
        frame.rename(columns = column_map, inplace = True)
        
        return frame[column_map.values()]

class Optimizer_NBA_DraftKings(Optimizer_NBA):
    def __init__(self, frame):
        self._salary_cap = 50000
        self._positions = [C, SG, PG, SF, PF, G, F, UTIL]
        super(Optimizer_NBA_DraftKings, self).__init__(frame)
        
    # Can we replace this with an API call?
    def _get_salaries(self):
        raise Exception("Need to implement DraftKings salaries")