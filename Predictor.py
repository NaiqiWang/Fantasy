# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 09:19:42 2015

@author: lshulgin
"""

from Constants import *

class Predictor(object):
    def predict(self):
        frame = self._predict_internal()
                                
        # Drop unnecessary columns
        output = frame[[COL_NAME, COL_POSITION, COL_TEAM, COL_POINTS]].copy()
        
        return output