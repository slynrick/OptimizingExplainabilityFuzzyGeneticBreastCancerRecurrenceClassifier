# -*- coding: utf-8 -*-
"""
Created by Fabio Cardoso
2022-10-08


This Package is a framework to handle the search for the optimum
fuzzy system to solve the problem.


"""
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import math
import numpy as np
from deap import algorithms, base, creator, tools
from sklearn.metrics import mean_squared_error
from fsas.FuzzySystemHandler import FuzzySystemHandler


class GeneticAlgorithmHandler:
    """Class to construct and handle the genetic algorithm

    Parameters
    ----------
    msg : str
        Human readable string describing the exception.
    code : :obj:`int`, optional
        Numeric error code.

    Attributes
    ----------
    msg : str
        Human readable string describing the exception.
    code : int
        Numeric error code.

    """

    def __init__(self, fuzzy_system: FuzzySystemHandler, max_rules: int = 200):
        self.__toolbox: base.Toolbox = base.Toolbox()
        self.__base_fuzzy_system: FuzzySystemHandler = fuzzy_system
        self.__max_rules: int = max_rules

        self.__individual_gen_per_rules: int = 0
        self.__low: list[int] = []
        self.__up: list[int] = []
        self.__setup_ga()
    
    def unpack_individual(self, individual) -> Dict[str, Any]:
        individual_dict: Dict[str, Any] = {
            'rules': [],
            'fs_conf': {
                'defuzz': 0,
                'sets': []
            }
        }

        individual_rules_part = individual[:self.__max_rules*self.__individual_gen_per_rules]
        for i in range(0, len(individual_rules_part), self.__individual_gen_per_rules):
            rule_list: List[Any] = individual_rules_part[i: i + self.__individual_gen_per_rules]
            rule = {
                'Enable': 1,
                'Consequent': rule_list.pop(),
                'Antecedents': rule_list
            }
            rule['Enable'] = 0 if all([a == 0 for a in rule['Antecedents']]) else 1
            individual_dict['rules'].append(rule)

        individual_fs_part = individual[self.__max_rules*self.__individual_gen_per_rules:]

        idx = 0
        for _, nsets in zip(self.__base_fuzzy_system.get_feature_columns() + [self.__base_fuzzy_system.get_target_column()], self.__base_fuzzy_system.get_feature_sets() + [self.__base_fuzzy_system.get_target_nsets()]):
            set_point = individual_fs_part[idx:idx+nsets]
            set_point = [0.] + set_point + [1.]

            individual_dict['fs_conf']['sets'].append(set_point)
            idx += nsets

        return individual_dict

    
    def __create_individual(self) -> None:
        #Define antecedents
        individual_rules_part = [] # Enable, Antecedents, Consequents
        for feature, nsets in zip(self.__base_fuzzy_system.get_feature_columns(), self.__base_fuzzy_system.get_feature_sets()):
            attr_name: str = f'attr_ant_{feature}'
            self.__toolbox.register(attr_name, random.randint, 0, nsets)
            individual_rules_part += [self.__toolbox.__dict__[attr_name]]

            self.__low += [0]
            self.__up += [nsets]

        #Define consequent
        attr_name: str = f'attr_cons_{self.__base_fuzzy_system.get_target_column()}'
        self.__toolbox.register(attr_name, random.randint, 0, self.__base_fuzzy_system.get_n_classes()-1)
        individual_rules_part += [self.__toolbox.__dict__[attr_name]]
        self.__low += [0]
        self.__up += [self.__base_fuzzy_system.get_n_classes()-1]
        
        self.__individual_gen_per_rules = len(individual_rules_part)
        individual_rules_part = individual_rules_part * self.__max_rules
        self.__low = self.__low * self.__max_rules
        self.__up = self.__up * self.__max_rules
        

        individual_fs_part = []
        # Define sets
        individual_fs_sets_part = []
        for feature, nsets in zip(self.__base_fuzzy_system.get_feature_columns(), self.__base_fuzzy_system.get_feature_sets()):
            attr_name: str = f'attr_ant_sets_{feature}'
            self.__toolbox.register(attr_name, random.uniform, 0, 1)
            individual_fs_sets_part += [self.__toolbox.__dict__[attr_name]] * nsets

            self.__low += [0.] * nsets
            self.__up += [1.] * nsets
        
        attr_name: str = f'attr_cons_sets_{self.__base_fuzzy_system.get_target_column()}'
        self.__toolbox.register(attr_name, random.uniform, 0, 1)
        individual_fs_sets_part += [self.__toolbox.__dict__[attr_name]] * self.__base_fuzzy_system.get_target_nsets()
        self.__low += [0.] * self.__base_fuzzy_system.get_target_nsets()
        self.__up += [self.__base_fuzzy_system.get_n_classes()-1] * self.__base_fuzzy_system.get_target_nsets()
        
        individual_fs_part += individual_fs_sets_part

        self.__toolbox.register('individual', tools.initCycle, creator.Individual, tuple(individual_rules_part + individual_fs_part), 1)

    def __setup_ga(self) -> None:
        random.seed(time.time())
        creator.create('FitnessMin', base.Fitness, weights=(-1.0, ))
        creator.create('Individual', list, fitness=creator.FitnessMin)

        self.__create_individual()
        self.__toolbox.register('population', tools.initRepeat, list, self.__toolbox.individual)

    
    def custom_mutation(self, individual, eta, indpb, low, up):
        x = tools.mutPolynomialBounded(individual, eta, low, up, indpb)
        new_individual = x[0]
        for g, x in zip(range(len(new_individual)), low):
            if isinstance(x, int):
                new_individual[g] = int(new_individual[g])
        return new_individual,
    
    def custom_cross(self, individual1, individual2, eta, low, up):
        x_ind = tools.cxSimulatedBinaryBounded(individual1, individual2, eta, low, up)
        new_individual = []
        for i in x_ind:
            for g, x in zip(range(len(i)), low):
                if isinstance(x, int):
                    i[g] = int(i[g])
            new_individual.append(i)
        return tuple(new_individual)
    
    def selElitistAndTournament(self, individuals, k, frac_elitist, tournsize):
        return tools.selBest(individuals, int(k*frac_elitist)) + tools.selTournament(individuals, int(k*(1-frac_elitist)), tournsize=tournsize)

    def get_individual_data(self, individual: List[Any]) -> List[int]:
        individual_unpacked: Dict[str, Any] = self.unpack_individual(individual)
        num_invalid_rules = 0
        num_repeated_rules = 0
        enable_rules = 0

        slots_used = []

        rules_per_class = [0] * self.__base_fuzzy_system.get_n_classes()

        rule_check = {}
        for rule in individual_unpacked['rules']:
            if int(rule['Enable']) == 1:
                invalid_rule = False
                enable_rules += 1
                ant = [x == 0 for x in rule['Antecedents']]
                slots_used.append(len([x for x in ant if x == False]))
                hash_ant = hash(tuple(rule['Antecedents']))
                if hash_ant in rule_check:
                    if rule_check[hash_ant] != rule['Consequent']:
                        invalid_rule = True
                    else:
                        num_repeated_rules += 1
                else:
                    rule_check[hash_ant] = rule['Consequent']
                num_invalid_rules += 1 if invalid_rule else 0
                rules_per_class[rule['Consequent']] += 1
        
        total_pair = 0
        invalid_pair = 0
        for sets in individual_unpacked['fs_conf']['sets']:
            list_points = sets
            for s, t in zip(range(len(list_points)), range(1, len(list_points))):
                if list_points[s] > list_points[t]:
                    list_points[t] = list_points[s]
                    invalid_pair += 1
                total_pair += 1
        
        return [enable_rules, rules_per_class, num_invalid_rules, num_repeated_rules, len(self.__base_fuzzy_system.get_feature_columns()), slots_used, total_pair, invalid_pair]


    def objective_function(self, individual: List[Any], print_all: bool = False) -> float:
        individual_unpacked: Dict[str, Any] = self.unpack_individual(individual)
        self.__base_fuzzy_system.set_new_sets(individual_unpacked)
        self.__base_fuzzy_system.set_rules_system(individual_unpacked)        
        acc, fail_to_predict = self.__base_fuzzy_system.evaluate_system()

        enable_rules, rules_per_class, num_invalid_rules, num_repeated_rules, max_slots_to_use, slots_used, total_pair, invalid_pair = self.get_individual_data(individual)
        
        perc_invalid_pair = invalid_pair/total_pair
        
        # Invalid rules percentual: num_invalid_rules/enable_rules
        invalid_perc = num_invalid_rules/enable_rules if enable_rules > 0 else 0

        # Repeated rules percentual: num_repeated_rules/enable_rules
        repeated_rules_perc = num_repeated_rules/enable_rules if enable_rules > 0 else 0

        # Average of antecedents slots used: np.average(slots_used)/len(self.__base_fuzzy_system.get_feature_columns())
        mean_slots_used = 0 if len(slots_used) == 0 else np.average(slots_used)
        slots_used_perc = mean_slots_used/max_slots_to_use

        enable_rules_perc = enable_rules/self.__max_rules
        
        if print_all:
            print(f"acc={acc} enable_rules_perc={enable_rules_perc} rules_per_class={rules_per_class} fail_to_predict={fail_to_predict} invalid_perc={invalid_perc} repeated_rules_perc={repeated_rules_perc} slots_used_perc={slots_used_perc} perc_invalid_pair={perc_invalid_pair}")

        target_enable_rules_perc = self.__base_fuzzy_system.get_n_classes()/self.__max_rules
        target_slots_used_perc = 1 / len(self.__base_fuzzy_system.get_feature_columns())
        target =  [    3,               0, target_enable_rules_perc, target_slots_used_perc,            0,                   0,                 0]
        current = [3*acc, fail_to_predict,        enable_rules_perc,        slots_used_perc, invalid_perc, repeated_rules_perc, perc_invalid_pair]

        objective = mean_squared_error(target, current)

        return (objective, )

    def feasible(self, individual: List[Any]) -> bool:
        enable_rules, rules_per_class, num_invalid_rules, num_repeated_rules, max_slots_to_use, slots_used, total_pair, invalid_pair = self.get_individual_data(individual)

        if any([x == 0 for x in slots_used]): # search for a simple rule: at least 1 antecedent
            return False
        if enable_rules < self.__base_fuzzy_system.get_n_classes():
            return False
        if all([x == 0 for x in rules_per_class]):
            return False
        if num_invalid_rules > 0 or num_repeated_rules > 0:
            return False
        if invalid_pair > 0:
            return False

        return True

    def distance_func(self, individual: List[Any]) -> float:
        enable_rules, rules_per_class, num_invalid_rules, num_repeated_rules, max_slots_to_use, slots_used, total_pair, invalid_pair = self.get_individual_data(individual)
        
        perc_invalid_pair = invalid_pair/total_pair
        
        # Invalid rules percentual: num_invalid_rules/enable_rules
        invalid_perc = num_invalid_rules/enable_rules if enable_rules > 0 else 0

        # Repeated rules percentual: num_repeated_rules/enable_rules
        repeated_rules_perc = num_repeated_rules/enable_rules if enable_rules > 0 else 0

        # Average of antecedents slots used: np.average(slots_used)/len(self.__base_fuzzy_system.get_feature_columns())
        mean_slots_used = 0 if len(slots_used) == 0 else np.average(slots_used)
        slots_used_perc = mean_slots_used/max_slots_to_use

        enable_rules_perc = enable_rules/self.__max_rules

        target_enable_rules_perc = self.__base_fuzzy_system.get_n_classes()/self.__max_rules
        target_slots_used_perc = 1 / len(self.__base_fuzzy_system.get_feature_columns())
        target =  [target_enable_rules_perc, target_slots_used_perc,            0,                   0,                 0]
        current = [       enable_rules_perc,        slots_used_perc, invalid_perc, repeated_rules_perc, perc_invalid_pair]

        objective = mean_squared_error(target, current)

        # np.sum([enable_rules_perc, (1-val_per_class), invalid_perc, repeated_rules_perc, slots_used_perc, perc_invalid_pair])
        return np.sum([invalid_perc, repeated_rules_perc, slots_used_perc, perc_invalid_pair])
        
    def run(self, cxpb: float = 0.5, mutpb: float = 0.1, npop: int = 100, ngen: int = 100) -> Tuple[List[Any], Any]:
        self.__toolbox.register('evaluate', self.objective_function)
        # self.__toolbox.decorate('evaluate', tools.DeltaPenalty(self.feasible, 1))#, self.distance_func))
        self.__toolbox.register('mate', self.custom_cross, eta=0.5, low=self.__low, up=self.__up)
        self.__toolbox.register('mutate', self.custom_mutation, eta=0.5, low=self.__low, up=self.__up, indpb=0.05)
        self.__toolbox.register("select", self.selElitistAndTournament, frac_elitist=0.1 , tournsize=3)
        pop = self.__toolbox.population(n=npop)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('avg', np.mean)
        stats.register('std', np.std)
        stats.register('min', np.min)
        stats.register('max', np.max)

        pop, log = algorithms.eaSimple(population=pop, toolbox=self.__toolbox, cxpb=cxpb, mutpb=mutpb,
                               ngen=ngen, stats=stats, halloffame=hof, verbose=True)
        return hof[0], pop
        