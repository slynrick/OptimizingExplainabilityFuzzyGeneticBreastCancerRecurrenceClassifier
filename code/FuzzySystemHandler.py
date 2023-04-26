# -*- coding: utf-8 -*-
"""
Created by Fabio Cardoso
2022-10-08


This Package is a framework to handle the search for the optimum
fuzzy system to solve the problem.


"""
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


class FuzzySystemHandler:
    def __init__(self, data: pd.DataFrame, feature_columns_sets: List[Tuple[str, int]], target_column: str, target_nsets: int, scaler: MinMaxScaler = None):
        self.__data: pd.DataFrame = data
        
        self.__feature_columns: List[str] = list(map(lambda x: x[0], feature_columns_sets))
        self.__feature_sets: List[int] = list(map(lambda x: x[1], feature_columns_sets))
        self.__target_column: str = target_column
        self.__target_nsets: int = target_nsets
        self.__n_classes: int = self.__target_nsets

        self.__normalized_data: pd.DataFrame = data.copy()
        X = self.__normalized_data[self.__feature_columns].to_numpy()
        y = self.__normalized_data[self.__target_column].to_numpy()
        self.__scaler = MinMaxScaler()
        self.__scaler.fit(X)
        if scaler is not None:
            self.__scaler = scaler
        self.__normalized_data[self.__feature_columns] = self.__scaler.transform(X)



        self.__fuzzy_system: Dict[str, Any] = {
            'variables': [],
            'rules': [],
            'implication': np.fmin,
            'aggregation': np.fmax,
            'defuzz': 'centroid'
        }
    def get_scaler(self) -> MinMaxScaler:
        return self.__scaler

    def get_data(self)-> pd.DataFrame:
        return self.__data
    
    def get_normalized_data(self) -> pd.DataFrame:
        return self.__normalized_data
    
    def get_feature_columns(self) -> List[str]:
        return self.__feature_columns
    
    def get_feature_sets(self) -> List[int]:
        return self.__feature_sets
    
    def get_target_column(self) -> str:
        return self.__target_column
    
    def get_target_nsets(self) -> List[int]:
        return self.__target_nsets
    
    def get_n_classes(self) -> int:
        return self.__n_classes

    def set_new_sets(self, individual: Dict[str, Any])-> None:
        self.__fuzzy_system['variables'] = []

        id = 0
        for feature, _ in zip(self.__feature_columns + [self.__target_column], self.__feature_sets + [self.__target_nsets]):
            universe = np.linspace(0, 1, 100)
            
            # just to pass through if any problem

            list_points = individual['fs_conf']['sets'][id]
            for s, t in zip(range(len(list_points)), range(1, len(list_points))):
                if list_points[s] > list_points[t]:
                    list_points[t] = list_points[s]


            sets = []
            v = list(range(len(list_points)-2))
            for i in v:
                membership = fuzz.trimf(universe, [list_points[i], list_points[i+1], list_points[i+2]])
                if i == v[0]:
                    membership = fuzz.trapmf(universe, [list_points[i], list_points[i], list_points[i+1], list_points[i+2]])
                if i == v[-1]:
                    membership = fuzz.trapmf(universe, [list_points[i], list_points[i+1], list_points[i+2], list_points[i+2]])
                sets.append({
                    'id': i,
                    'name': f'{feature}_set{i}',
                    'membership': membership
                }) 

            f = {
                'id': id,
                'name': feature,
                'universe': universe,
                'sets': sets
            }
            id += 1

            self.__fuzzy_system['variables'].append(f)


    def set_rules_system(self, individual: Dict[str, Any]) -> None:
        rules =[]
        for rule in individual['rules']:
            if int(rule['Enable']) == 0:
                continue
            antecedents = []
            for index, cset in enumerate(rule['Antecedents']):
                if cset > 0:
                    variable: Dict[str, Any] = self.__fuzzy_system['variables'][index]
                    antecedents.append((variable, int(cset)))
                else:
                    antecedents.append((None, cset))
            
            variable: Dict[str, Any] = self.__fuzzy_system['variables'][-1]

            rules.append({
                'antecedents': antecedents,
                'consequent': (variable, int(rule['Consequent']))
            })
        self.__fuzzy_system['rules'] = rules

    def evaluate_system(self) -> Tuple[float, float]:
        antecedent_f = self.__fuzzy_system['implication']

        fail_to_predict = 0
        y_pred = []
        for _, row in self.__normalized_data.iterrows():
            activated_rules = []
            for rule in self.__fuzzy_system['rules']:
                anteced_values= []
                if any([cset > 0 for _, cset in rule['antecedents']]): #valid rules
                    for idx, f in enumerate(self.__feature_columns):
                        variable, cset = rule['antecedents'][idx]
                        if cset > 0:
                            variable_set = variable['sets'][cset-1]['membership']
                            variable_universe = variable['universe']
                            anteced_values.append(fuzz.interp_membership(variable_universe, variable_set, row[f]))
                    c_variable, c_cset = rule['consequent']
                    c_variable_id = c_variable['sets'][c_cset]['id']
                    
                    active_r = 0
                    if len(anteced_values) > 0:
                        if len(anteced_values) == 1:
                            active_r = anteced_values[-1]
                        elif len(anteced_values) >= 2:
                            active_r = antecedent_f(anteced_values[0], anteced_values[1])
                            for i in range(2, len(anteced_values)):
                                active_r = antecedent_f(active_r, anteced_values[i])
                    activated_rules.append((active_r, c_variable_id))
            
            classes = [0.] * self.__n_classes
            for p, c in activated_rules:
                classes[c] = max(classes[c], p)
        
            index_max = np.argmax(classes)
            if max(classes) == 0:
                fail_to_predict += 1
                y_pred.append(-1)
            else:
                y_pred.append(index_max)

        return accuracy_score(self.__normalized_data[self.__target_column].tolist(), y_pred), fail_to_predict/len(y_pred)
    
    def compute_result(self, data: pd.DataFrame) -> List[float]:
        antecedent_f = self.__fuzzy_system['implication']

        normalized_data: pd.DataFrame = data.copy()
        X = normalized_data[self.__feature_columns].to_numpy()
        y = normalized_data[self.__target_column].to_numpy()
        normalized_data[self.__feature_columns] = self.__scaler.transform(X)

        y_pred = []
        for _, row in normalized_data.iterrows():
            activated_rules = []
            for rule in self.__fuzzy_system['rules']:
                anteced_values= []
                if any([cset > 0 for _, cset in rule['antecedents']]): #valid rules
                    for idx, f in enumerate(self.__feature_columns):
                        variable, cset = rule['antecedents'][idx]
                        if cset > 0:
                            variable_set = variable['sets'][cset-1]['membership']
                            variable_universe = variable['universe']
                            anteced_values.append(fuzz.interp_membership(variable_universe, variable_set, row[f]))
                    c_variable, c_cset = rule['consequent']
                    c_variable_id = c_variable['sets'][c_cset]['id']
                    
                    active_r = 0
                    if len(anteced_values) > 0:
                        if len(anteced_values) == 1:
                            active_r = anteced_values[-1]
                        elif len(anteced_values) >= 2:
                            active_r = antecedent_f(anteced_values[0], anteced_values[1])
                            for i in range(2, len(anteced_values)):
                                active_r = antecedent_f(active_r, anteced_values[i])
                    activated_rules.append((active_r, c_variable_id))
            
            classes = [0.] * self.__n_classes
            for p, c in activated_rules:
                classes[c] = max(classes[c], p)
        
            index_max = np.argmax(classes)
            if max(classes) == 0:
                y_pred.append(-1)
            else:
                y_pred.append(index_max)

        return y_pred