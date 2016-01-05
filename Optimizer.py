# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 16:46:16 2015

@author: lshulgin
"""

from Constants import *
import pandas as pd
from pulp import *
from Roster import Roster

class Optimizer(object):
    def __init__(self, model_output):
        self._frame = model_output.copy()
        self._prob = None

    def setup(self):
        self.decorate_data()
        
        self._prob = LpProblem("Fantasy", LpMaximize)

        self._players = list(self._frame.index)
        self._choices = LpVariable.dict("Include", self._players, 0, 1, LpInteger)
        
        # Function to maximize
        self._prob += lpSum([self._choices[player] * self._frame[COL_POINTS][player] for player in self._players]), "Projected points"
        
        # Position constraints
        constraints = self.get_constraints()
        for key, data in constraints.iteritems():
            self._prob += lpSum([self._choices[player] * self._frame[key][player] for player in self._players]) >= data['Min'], key + " Min"
            self._prob += lpSum([self._choices[player] * self._frame[key][player] for player in self._players]) <= data['Max'], key + " Max"
  
        # Salary constraint
        self._prob += lpSum([self._choices[player] * self._frame[COL_SALARY][player] for player in self._players]) <= self._salary_cap, "Salary cap"

        
    def decorate_data(self):
        self._frame[COL_USED] = 0

        for pos in set(self._positions):
            if pos in self._position_map:            
                self._frame[pos] = map(lambda position: 1 if position in self._position_map[pos] else 0, self._frame[COL_POSITION])
            else:
                self._frame[pos] = map(lambda position: 1 if position == pos else 0, self._frame[COL_POSITION])
        
        salaries = self._get_salaries()
        
        # Now need to join self._frame and salaries using a proper key, which isn't implemented yet!
        # For now use [Name, Position, Team], which isn't great because the two name columns
        # come from different sources
        self._frame = pd.merge(self._frame, salaries, 'left', [COL_NAME, COL_POSITION, COL_TEAM])


    # Take an array of avaiable positions and convert into min/max counts for each tag
    def get_constraints(self):
        keys = set(self._positions)
        constraints = {}        
        
        for key in keys:
            min_times = 0
            max_times = 0
            for entry in self._positions:
                if entry == key:
                    min_times += 1
                    max_times += 1
                elif entry in self._position_map and key in self._position_map[entry]:
                    max_times += 1
                elif key in self._position_map and entry in self._position_map[key]:
                    min_times += 1
                    max_times += 1
            constraints[key] = {'Min' : min_times, 'Max' : max_times}
        
        return constraints

    
    def disjoint_rosters(self, n):
        self.setup()
                        
        rosters = []
        while(len(rosters) < n):
            self._prob.solve()
            
            if LpStatus[self._prob.status] != 'Optimal':
                raise "Unable to find optimal solution"
            
            players = filter(lambda x: value(self._choices[x]) == 1, self._choices)
            valid = self._frame.index.map(lambda x: x in players)
            roster = Roster(self._frame[valid].copy())
            rosters.append(roster)
            
            # Exclude these player from further iterations
            for player in players:
                if self._frame[COL_USED][player] == 0:
                    self._frame.set_value(player, COL_USED, 1)
                    self._prob += self._choices[player] == 0, "Exclude " + str(player)
        
        return rosters

    # Get the n top rosters -- this is broken since only returns one roster from each degenerate score    
    def top_rosters(self, n):
        self.setup()
              
        # For some reason smaller steps like 1e-6 don't work
        epsilon = .001
            
        count = 0
        while(count < n):
            self._prob.solve()
            
            if LpStatus[self._prob.status] != 'Optimal':
                raise "Unable to find optimal solution"
            
            players = filter(lambda x: value(self._choices[x]) == 1, self._choices)
            valid = self._frame.index.map(lambda x: x in players)
            roster = Roster(self._frame[valid].copy())
            yield roster
            count += 1
            
            # Set the threshold a bit lower
            high_score = value(self._prob.objective)
            self._prob += lpSum([self._choices[player] * self._frame[COL_POINTS][player] for player in self._players]) <= high_score - epsilon, "{0}th highest".format(count)