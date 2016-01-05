# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 15:47:28 2015

@author: lshulgin
"""

from Constants import *


class Roster(object):
    def __init__(self, frame):
        self._frame = frame
    
    def __repr__(self):
        temp_frame = self._frame.copy()
        temp_frame.sort([COL_POSITION], inplace = True)               
        
        output = ""         
        output += repr(temp_frame[[COL_POSITION, COL_NAME, COL_TEAM, COL_POINTS, COL_SALARY]])
        output += '\n\n'
        output += 'Projected score: {0}\n'.format(temp_frame[COL_POINTS].sum())
        output += 'Salary: ${:,}\n'.format(temp_frame[COL_SALARY].sum())
        
        return output