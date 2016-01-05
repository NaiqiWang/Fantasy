# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 23:40:11 2015

@author: lshulgin
"""

from Optimizer_NBA import *
from Optimizer_NFL import *
from Predictor_NBA import *
from Predictor_NFL import *


predictor = Predictor_NBA_FanDuel_Dummy()
#predictor = Predictor_NFL_DraftKings_Dummy()
#predictor = Predictor_NFL_FanDuel_Dummy()


model_output = predictor.predict()


#optimizer = Optimizer_NBA_DraftKings(model_output)
optimizer = Optimizer_NBA_FanDuel(model_output)
#optimizer = Optimizer_NFL_DraftKings(model_output)
#optimizer = Optimizer_NFL_FanDuel(model_output)
	

#rosters = optimizer.disjoint_rosters(3)
rosters = optimizer.top_rosters(3)


for i, roster in enumerate(rosters):
    print "Roster #{0}:\n\n{1}\n".format(i + 1, roster)