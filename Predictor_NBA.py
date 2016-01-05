# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 15:43:01 2015

@author: lshulgin
"""

from Constants import *
import os
import pandas as pd
from Predictor import Predictor

class Predictor_NBA(Predictor):
    def load_historical_data(self):
        filename = os.path.join(DIR_STATS, 'All_Players_Log.csv')
        
        frame = pd.read_csv(filename)
                                    
        frame[COL_NAME] = map(_convert_name, frame["Player"])
        
        return frame
        

class Predictor_NBA_FanDuel_Dummy(Predictor_NBA):        
    def _predict_internal(self):
        # Just use raw site output for now
        filename = os.path.join(DIR_SALARIES, 'FanDuel NBA Output 25Dec15.csv')
        
        frame = pd.read_csv(filename)
        frame[COL_NAME] = map(lambda first, last: "{0} {1}".format(first, last), frame['First Name'], frame['Last Name'])        
        
        # Drop injured players -- overly aggressive here!
        frame = frame[pd.isnull(frame["Injury Indicator"])]
        
        column_map = {
            COL_NAME : COL_NAME,
            'Position' : COL_POSITION,
            'Team' : COL_TEAM,
            'FPPG' : COL_POINTS,
        }
        
        frame.rename(columns = column_map, inplace = True)
        
        return frame[column_map.values()]
        
    def _convert_name(lastfirst):
        parts = lastfirst.split(",")
        if len(parts) > 2:
            raise "Name '{0}' has more than one comma!".format(lastfirst)
        if len(parts) == 1: # this case for Nene, etc.
            return lastfirst
        
        last, first = parts[0].strip(), parts[1].strip()
        return "{0} {1}".format(first, last)