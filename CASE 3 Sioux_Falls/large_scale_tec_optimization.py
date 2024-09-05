import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
import csv
import time

import tensorflow as tf
print(tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs available:", gpus)
else:
    print("No GPUs were found.")
# show all columns
pd.set_option('display.max_columns', None)

# ==================== Global variables ====================
# global params
# global g_zone_list
# global g_demand_list
# global g_zone_seq_id_dict
# global g_zone_id_seq_dict
# global variable_list
# global g_zone_id_agency_id_dict
g_variable_list_dict = {}
g_gradient_list_dict = {}

g_agency_list = []
g_agency_dict = {}
g_zone_list = []
g_demand_list = []
g_zone_seq_id_dict = {}
g_zone_id_seq_dict = {}
params = {}
g_zone_id_agency_id_dict = {}
g_agency_id_zone_id_dict = {}
var_dict = {}
var_upper_bound = {}
log_flag = True
g_tensor_seq_dict = {}

g_od_group_zone_group_dict = {}
g_od_group_agency_dict = {}
# equity_method = 'Gini'
# equity_method = 'Coefficient of Variation'
# equity_method = 'Atkinson'
equity_method = 'Generalized Entropy(0)'


# optimizer = SGD(learning_rate=0.001, momentum=0.9)
# ==================== Class definition ====================
class Agency:
    def __init__(self, agency_id, agency_name):
        self.agency_id = agency_id
        self.agency_name = agency_name
        self.pop = 0
        self.mean_vot = 0
        self.total_trips = 0
        self.base_revenue = 0
        self.loss_func = 0
        self.prev_loss = - 0.1

        self.equity_improvement = None
        self.adjusted_equity_index = None
        self.equity_gap = None
        self.tec_demand = None
        self.equity_income = None
        self.total_subsidy = None
        self.adjusted_transit_income = None
        self.equity_change_income = None
        self.adjusted_transit_revenue = None
        self.mean_accessibility = None

        self.agent_zone_group_trip_tensor = None

        self.group_pop = {}
        self.zone_list = []
        self.od_list = []
        self.var_dict = {}
        self.var_upper_bound = {}

    def calculate_agency_total_population(self):
        self.pop = 0
        for zone_obj in self.zone_list:
            self.pop += zone_obj.pop
        print("agency:", self.agency_id, " pop:", self.pop)

    def calculate_agency_group_population(self):
        for zone_obj in self.zone_list:
            for g_id in zone_obj.zone_group_list:
                if g_id in self.group_pop:
                    self.group_pop[g_id] += zone_obj.zone_pop_dict[g_id]
                else:
                    self.group_pop[g_id] = zone_obj.zone_pop_dict[g_id]
        for g_id in self.group_pop.keys():
            print("agency:", self.agency_id, " group:", g_id, " group_pop:", self.group_pop[g_id])

    def calculate_agency_mean_vot(self):
        for zone_obj in self.zone_list:
            zone_pop_ratio_agency = zone_obj.pop / self.pop
            for g_id in zone_obj.zone_group_list:
                self.mean_vot += (zone_pop_ratio_agency * zone_obj.zone_pop_ratio_dict[g_id] *
                                  zone_obj.zone_group_vot_dict[g_id])
        print("agency:", self.agency_id, " mean_vot:", self.mean_vot)

    def calculate_agency_total_trips(self):
        for od in self.od_list:
            self.total_trips += od.trips
        print("agency:", self.agency_id, " trips:", self.total_trips)


class Zone:
    def __init__(self, zone_id, zone_seq, x, y, geometry, pop, zone_group_list, zone_group_pop_dict,
                 zone_group_pop_ratio_dict, zone_group_income_dict, zone_group_hourly_pay_dict,
                 zone_group_vot_dict, agency_id):
        self.zone_id = zone_id
        self.zone_seq = zone_seq
        self.x = x
        self.y = y
        self.geometry = geometry
        self.pop = pop
        # percentage of population for each zone in the total population
        self.pop_ratio = None  # to be calculated
        self.trip_ratio = None  # to be calculated
        self.zone_group_list = zone_group_list
        self.zone_group_dict = {}
        self.zone_pop_dict = zone_group_pop_dict
        # percentage of population of each group in the total population of the zone
        self.zone_pop_ratio_dict = zone_group_pop_ratio_dict
        # income of each group in the zone
        self.zone_income_dict = zone_group_income_dict
        # hourly pay of each group in the zone (i.e., value of time)
        self.zone_hourly_pay_dict = zone_group_hourly_pay_dict
        self.zone_group_vot_dict = zone_group_vot_dict
        self.agency_id = agency_id
        self.od_list = []
        self.zone_total_trips = None
        self.initialize()

    def initialize(self):
        for grp_id in self.zone_group_list:
            zone_group = ZoneGroup(self.zone_id, grp_id, self.zone_pop_dict[grp_id],
                                   self.zone_pop_ratio_dict[grp_id], self.zone_hourly_pay_dict[grp_id],
                                   self.zone_income_dict[grp_id], self.zone_group_vot_dict[grp_id], self.agency_id)
            self.zone_group_dict[grp_id] = zone_group


class ZoneGroup:
    def __init__(self, zone_id, grp_id, pop, pop_ratio, hourly_pay, income, vot, agency_id):
        self.zone_id = zone_id
        self.group_id = grp_id
        self.zone_group_id = 100 * zone_id + grp_id
        self.pop = pop
        self.pop_ratio = pop_ratio
        self.hourly_pay = hourly_pay
        self.income = income
        self.vot = vot
        self.agency_id = agency_id
        # self.accessibility = None
        # self.pop_weighted_accessibility = None
        # self.group_total_trips = None
        # self.group_trips_ratio = None
        # self.base_entropy = None
        # self.theil_index = None
        # self.adjusted_accessibility = None
        # self.adjusted_pop_weighted_accessibility = None
        # self.adjusted_entropy = None
        # self.adjusted_theil_index = None
        # self.agency_pop_weighted_accessibility = None
        # self.adjusted_agency_pop_weighted_accessibility = None
        # self.agency_base_entropy = None
        # self.agency_theil_index = None
        # self.agency_adjusted_entropy = None
        # self.agency_adjusted_theil_index = None


# demand class
class Demand:
    def __init__(self, demand_id, od_seq, from_zone_id, to_zone_id, trips, od_group_dict, od_group_list,
                 source_zone, agency_id):
        self.demand_id = demand_id
        self.od_seq = od_seq
        self.from_zone_id = from_zone_id
        self.to_zone_id = to_zone_id
        self.trips = trips
        self.od_group_dict = od_group_dict
        self.od_group_list = od_group_list
        self.source_zone = source_zone
        self.agency_id = agency_id


class ODGroup:
    def __init__(self, od_group_id, od_group_seq, demand_id, group_id, from_zone_id, to_zone_id, trips, pop_ratio,
                 group_trips, auto_tc, auto_tt, transit_tc, transit_tt, od_group_hourly_pay, od_group_vot, agency_id):

        self.od_group_id = od_group_id
        self.od_group_id = od_group_seq
        self.demand_id = demand_id
        self.group_id = group_id
        self.from_zone_id = from_zone_id
        self.to_zone_id = to_zone_id
        self.trips = trips
        self.pop_ratio = pop_ratio
        self.group_trips = group_trips
        self.auto_tc = auto_tc
        self.auto_tt = auto_tt
        self.transit_tc = transit_tc  # set the maximum value of transit_tc to 3 according to sioux_falls
        self.transit_tt = transit_tt
        self.od_group_hourly_pay = od_group_hourly_pay
        self.od_group_vot = od_group_vot
        self.agency_id = agency_id
        self.base_auto_utility = None
        self.base_transit_utility = None
        self.base_auto_prob = None
        self.base_transit_prob = None
        self.base_od_transit_accessibility = None
        self.base_transit_volume = None
        self.base_auto_volume = None
        # variables
        self.initial_subsidy = 0
        self.transit_subsidy = None
        self.auto_subsidy = None
        self.adjusted_transit_utility = None
        self.adjusted_auto_utility = None
        self.adjusted_auto_prob = None
        self.adjusted_transit_prob = None
        self.adjusted_auto_volume = None
        self.adjusted_transit_volume = None
        self.adjusted_od_transit_accessibility = None
        self.adjusted_accessibility = None
        self.initialize_variables()

    def initialize_variables(self):
        # determine the optimal value of the transit subsidy (only consider total revenue and subsidy cost using
        # bisection method)
        # range of the variable is from [0, 1.5 * transit_tc]
        lower_bound = 0
        upper_bound = 1.0 * self.transit_tc
        condition_flag = True
        while condition_flag:
            new_bound = (lower_bound + upper_bound) / 2
            # the first derivative of the total revenue with respect to the transit subsidy
            auto_utility = -self.auto_tc - self.auto_tt * self.od_group_vot
            transit_utility = -self.transit_tc - self.transit_tt * self.od_group_vot + new_bound
            odd_auto = np.exp(auto_utility)
            odd_transit = np.exp(transit_utility)
            transit_prob = odd_transit / (odd_auto + odd_transit)
            income_derivative = self.group_trips * self.transit_tc * transit_prob * (1 - transit_prob)
            subsidy_derivative = self.group_trips * (transit_prob + new_bound * transit_prob * (1 - transit_prob))
            derivative = income_derivative - subsidy_derivative
            # print("derivative:", derivative)
            # this is an maximization problem
            if derivative > 0:
                lower_bound = new_bound
            else:
                upper_bound = new_bound

            if upper_bound - lower_bound < 1e-5:
                condition_flag = False
        self.initial_subsidy = (lower_bound + upper_bound) / 2
        print("od:", self.demand_id, " group:", self.group_id, " initial transit subsidy:", self.initial_subsidy)

    def subsidy(self):
        if self.group_id != 100:
            name = str((self.from_zone_id, self.to_zone_id, self.group_id))
            # set lower bound and upper bound of the variable
            self.transit_subsidy = tf.Variable(self.initial_subsidy, dtype=tf.float32, name=name,
                                               constraint=lambda x: tf.clip_by_value(x, 0, 10))
            g_agency_dict[self.agency_id].var_dict[name] = self.transit_subsidy
            g_agency_dict[self.agency_id].var_upper_bound[name] = 1.5 * self.transit_tc
            var_dict[name] = self.transit_subsidy
            var_upper_bound[name] = 1.0 * self.transit_tc
            self.auto_subsidy = tf.constant(0.0, dtype=tf.float32)
        else:
            self.transit_subsidy = tf.constant(0.0, dtype=tf.float32)
            self.auto_subsidy = tf.constant(0.0, dtype=tf.float32)
        return self.transit_subsidy, self.auto_subsidy

    def calculate_adjusted_utility(self):
        self.adjusted_auto_utility = self.base_auto_utility + self.auto_subsidy
        self.adjusted_transit_utility = self.base_transit_utility + self.transit_subsidy
        # if log_flag:
        #     print("od:", self.demand_id, " group:", self.group_id,
        #           " adjusted_auto_utility:", self.adjusted_auto_utility,
        #           " adjusted_transit_utility:", self.adjusted_transit_utility)

    def calculate_adjusted_prob(self):
        odd_auto = tf.math.exp(self.adjusted_auto_utility)
        odd_transit = tf.math.exp(self.adjusted_transit_utility)
        self.adjusted_auto_prob = odd_auto / (odd_auto + odd_transit)
        self.adjusted_transit_prob = odd_transit / (odd_auto + odd_transit)
        # if log_flag:
        #     print("od:", self.demand_id, " group:", self.group_id,
        #           " adjusted_auto_prob:", self.adjusted_auto_prob,
        #           " adjusted_transit_prob:", self.adjusted_transit_prob)

    def calculate_adjusted_volume(self):
        self.adjusted_auto_volume = self.adjusted_auto_prob * self.group_trips
        self.adjusted_transit_volume = self.adjusted_transit_prob * self.group_trips
        # if log_flag:
        #     print("od:", self.demand_id, " group:", self.group_id,
        #           " adjusted_auto_volume:", self.adjusted_auto_volume,
        #           " adjusted_transit_volume:", self.adjusted_transit_volume)

    def calculate_od_adjusted_transit_accessibility(self):
        # self.adjusted_od_transit_accessibility = (
        #     max(1e-5, 1 - (self.transit_tc + self.transit_tt * params['mean_vot'] - self.transit_subsidy)
        #         / self.od_group_hourly_pay))
        self.adjusted_od_transit_accessibility = (
            tf.math.maximum(1e-5, 1 - (self.transit_tc + self.transit_tt * params['mean_vot'] - self.transit_subsidy)
                            / self.od_group_hourly_pay))
        # if log_flag:
        #     print("od:", self.demand_id, " group:", self.group_id,
        #           " adjusted_od_transit_accessibility:", self.adjusted_od_transit_accessibility)


class ComputationalGraph:
    def __init__(self, zone_group_id_list, zone_id_list,
                 inner_zone_group_id_list, zone_agency_id_list,
                 zone_trip_ratio_list, zone_trip_ratio_agency_list,
                 zone_pop_ratio_list, zone_pop_list,
                 od_group_id_list, od_agency_id_list,
                 demand_id_list, from_zone_id_list,
                 to_zone_id_list, inner_od_group_id_list,
                 od_group_trip_list, auto_travel_cost_list,
                 auto_travel_time_list, transit_travel_cost_list,
                 transit_travel_time_list, vot_list,
                 hourly_salary_list, incidence_matrix, od_group_initial_subsidy_list):
        # others
        self.adjusted_zone_transit_accessibility = None
        self.TEC_price = None
        self.equity_income = None
        self.adjusted_total_revenue = None
        self.TEC_demand = None
        self.equity_gap = None
        self.equity_improvement = None
        self.adjusted_equity_metrics = None
        self.transit_subsidy_tensor = None
        self.od_group_base_revenue_tensor = None
        self.base_subsidy_cost = None
        self.base_income = None
        self.od_group_base_subsidy_cost_tensor = None
        self.base_revenue = None
        self.od_group_base_income_tensor = None
        self.base_cv = None
        self.base_equity_metrics = None
        self.base_av_mean = None
        self.base_atkinson_index = None
        self.base_gini_coefficient = None
        self.base_ge1_entropy_metrics = None
        self.base_entropy_tensor2 = None
        self.base_ge0_equity_metrics = None
        self.base_entropy_tensor = None
        self.zone_group_trip_ratio_tensor = None
        self.mean_accessibility = None
        self.base_trip_weighted_accessibility = None
        self.od_trip_ratio_in_zone = None
        self.zone_group_trips_tensor = None
        self.base_auto_prob = None
        self.base_transit_prob = None
        self.base_auto_volume = None
        self.base_transit_volume = None
        self.base_od_transit_accessibility = None
        self.base_auto_utility = None
        self.base_transit_utility = None
        self.base_auto_utility_mean_vot = None
        self.base_transit_utility_mean_vot = None
        self.base_zone_transit_accessibility = None

        # id list of od group
        self.zone_group_id_list = zone_group_id_list
        self.zone_id_list = zone_id_list
        self.inner_zone_group_id_list = inner_zone_group_id_list
        self.zone_agency_id_list = zone_agency_id_list
        self.od_group_id_list = od_group_id_list
        self.od_agency_id_list = od_agency_id_list
        self.demand_id_list = demand_id_list
        self.from_zone_id_list = from_zone_id_list
        self.to_zone_id_list = to_zone_id_list
        self.inner_od_group_id_list = inner_od_group_id_list

        # convert list to tensor
        self.zone_trip_ratio_tensor = \
            tf.reshape(tf.convert_to_tensor(zone_trip_ratio_list, dtype=tf.float32), [1, -1])
        self.zone_trip_ratio_agency_tensor = \
            tf.reshape(tf.convert_to_tensor(zone_trip_ratio_agency_list, dtype=tf.float32), [1, -1])
        self.zone_pop_ratio_tensor = \
            tf.reshape(tf.convert_to_tensor(zone_pop_ratio_list, dtype=tf.float32), [1, -1])
        self.zone_pop_tensor = \
            tf.reshape(tf.convert_to_tensor(zone_pop_list, dtype=tf.float32), [1, -1])
        self.od_group_trip_tensor = \
            tf.reshape(tf.convert_to_tensor(od_group_trip_list, dtype=tf.float32), [1, -1])
        self.total_trips = \
            params['total_trips']
        self.od_group_trip_ratio_tensor = \
            self.od_group_trip_tensor / self.total_trips
        self.auto_travel_cost_tensor = \
            tf.reshape(tf.convert_to_tensor(auto_travel_cost_list, dtype=tf.float32), [1, -1])
        self.auto_travel_time_tensor = \
            tf.reshape(tf.convert_to_tensor(auto_travel_time_list, dtype=tf.float32), [1, -1])
        self.transit_travel_cost_tensor = \
            tf.reshape(tf.convert_to_tensor(transit_travel_cost_list, dtype=tf.float32), [1, -1])
        self.transit_travel_time_tensor = \
            tf.reshape(tf.convert_to_tensor(transit_travel_time_list, dtype=tf.float32), [1, -1])
        self.vot_tensor = \
            tf.reshape(tf.convert_to_tensor(vot_list, dtype=tf.float32), [1, -1])
        self.hourly_salary_tensor = \
            tf.reshape(tf.convert_to_tensor(hourly_salary_list, dtype=tf.float32), [1, -1])
        self.incidence_matrix = \
            tf.convert_to_tensor(incidence_matrix, dtype=tf.float32)
        self.od_group_initial_subsidy_tensor = \
            tf.reshape(tf.convert_to_tensor(od_group_initial_subsidy_list, dtype=tf.float32), [1, -1])
        # calculate the base utility
        self.calculate_base_utility()
        # calculate the base utility with mean vot
        self.calculate_base_utility_mean_vot()
        # calculate the base probability
        self.calculate_base_transit_prob()
        # calculate the base volume
        self.calculate_base_volume()
        # calculate the base od accessibility
        self.calculate_base_accessibility()
        # calculate the base revenue
        self.calculate_base_revenue()
        # calculate the base equity index
        if equity_method == 'Generalized Entropy(0)':  # Theil0
            # calculate the ge0 equity index
            self.calculate_base_ge0_equity_index()
        elif equity_method == 'Generalized Entropy(1)':  # Theil1
            # calculate the ge0 equity index
            self.calculate_base_ge1_equity_index()
        elif equity_method == 'Gini':  # Gini
            self.calculate_base_gini_coefficient()
        elif equity_method == 'Coefficient of Variation':  # Coefficient of Variation
            self.calculate_base_cv()
        elif equity_method == 'Atkinson':  # Atkinson
            self.calculate_base_atkinson_index(0.5)
        else:
            self.calculate_base_ge0_equity_index()
        self.create_variables()

    def obtain_zone_group_agency_range(self, agency_id):
        # obtain the range of the zone group id for the agency
        index_list = []
        for index in range(len(self.zone_agency_id_list)):
            if self.zone_agency_id_list[index] == agency_id:
                index_list.append(self.zone_group_id_list[index])
        start = min(index_list)
        end = max(index_list) + 1
        return start, end

    def obtain_od_group_agency_range(self, agency_id):
        # obtain the range of the od group id for the agency
        index_list = []
        for index in range(len(self.od_agency_id_list)):
            if self.od_agency_id_list[index] == agency_id:
                index_list.append(self.od_group_id_list[index])
        start = min(index_list)
        end = max(index_list) + 1
        return start, end

    def obtain_unmask_index(self, agency_id):
        # obtain the unmask index for the agency
        unmask_list = []
        for index in range(len(self.od_agency_id_list)):
            if self.od_agency_id_list[index] == agency_id:
                unmask_list.append(True)
            else:
                unmask_list.append(False)
        unmask_tensor = tf.cast(unmask_list, dtype=tf.bool)
        return unmask_tensor

    def calculate_base_utility(self):
        # corresponding calculation the tensor or matrix
        self.base_auto_utility = \
            -self.auto_travel_cost_tensor - self.auto_travel_time_tensor * self.vot_tensor
        self.base_transit_utility = \
            -self.transit_travel_cost_tensor - self.transit_travel_time_tensor * self.vot_tensor

    def calculate_base_utility_mean_vot(self):
        self.base_auto_utility_mean_vot = -self.auto_travel_cost_tensor - self.auto_travel_time_tensor * params[
            'mean_vot']
        self.base_transit_utility_mean_vot = \
            -self.transit_travel_cost_tensor - self.transit_travel_time_tensor * params['mean_vot']

    def calculate_base_transit_prob(self):
        odd_auto = tf.math.exp(self.base_auto_utility)
        odd_transit = tf.math.exp(self.base_transit_utility)
        self.base_auto_prob = odd_auto / (odd_auto + odd_transit)
        self.base_transit_prob = odd_transit / (odd_auto + odd_transit)

    def calculate_base_volume(self):
        self.base_auto_volume = self.base_auto_prob * self.od_group_trip_tensor
        self.base_transit_volume = self.base_transit_prob * self.od_group_trip_tensor

    def calculate_base_accessibility(self):
        self.zone_group_trips_tensor = tf.matmul(self.od_group_trip_tensor, self.incidence_matrix)
        self.zone_group_trip_ratio_tensor = tf.matmul(self.od_group_trip_ratio_tensor, self.incidence_matrix)

        self.od_trip_ratio_in_zone = \
            self.od_group_trip_tensor / tf.transpose(
                tf.matmul(self.incidence_matrix, tf.transpose(self.zone_group_trips_tensor)))

        self.base_od_transit_accessibility = (
            tf.math.maximum(1e-5, 1 - (-self.base_transit_utility_mean_vot) / self.hourly_salary_tensor))
        self.base_od_transit_accessibility = self.base_od_transit_accessibility * self.od_trip_ratio_in_zone
        self.base_zone_transit_accessibility = tf.matmul(self.base_od_transit_accessibility, self.incidence_matrix)
        self.base_trip_weighted_accessibility = self.base_zone_transit_accessibility * self.zone_group_trip_ratio_tensor
        self.mean_accessibility = tf.reduce_sum(self.base_trip_weighted_accessibility)

        # calculate mean_accessibility for each agency
        params['mean_accessibility'] = self.mean_accessibility.numpy()
        for agen in g_agency_list:
            start_zg, end_zg = self.obtain_zone_group_agency_range(agen.agency_id)
            start_odg, end_odg = self.obtain_od_group_agency_range(agen.agency_id)
            agen.od_group_trip_tensor = self.od_group_trip_tensor[:, start_odg:end_odg]
            agen.incidence_matrix = self.incidence_matrix[start_odg:end_odg, start_zg:end_zg]
            agen.agent_zone_group_trip_tensor = tf.matmul(agen.od_group_trip_tensor, agen.incidence_matrix)
            agen.total_trips = tf.reduce_sum(agen.agent_zone_group_trip_tensor)
            agen.agency_trip_rate = agen.total_trips / self.total_trips
            params[str('agency_') + str(agen.agency_id) + '_total_trips'] = \
                tf.reduce_sum(agen.agent_zone_group_trip_tensor).numpy()
            agen.agent_zone_group_trip_ratio_tensor = agen.agent_zone_group_trip_tensor / agen.total_trips
            agen.base_zone_transit_accessibility = self.base_zone_transit_accessibility[:, start_zg:end_zg]
            agen.base_trip_weighted_accessibility = \
                agen.agent_zone_group_trip_ratio_tensor * agen.base_zone_transit_accessibility
            agen.mean_accessibility = tf.reduce_sum(agen.base_trip_weighted_accessibility)
            params[str('agency_') + str(agen.agency_id) + '_mean_accessibility'] = agen.mean_accessibility.numpy()

    def calculate_base_ge0_equity_index(self):
        # calculate the base entropy
        self.base_entropy_tensor = tf.math.log(self.mean_accessibility / self.base_zone_transit_accessibility)
        self.base_ge0_equity_metrics = tf.reduce_sum(self.base_entropy_tensor * self.zone_group_trip_ratio_tensor)
        for agen in g_agency_list:
            agen.base_entropy_tensor = tf.math.log(agen.mean_accessibility / agen.base_zone_transit_accessibility)
            agen.within_weight_ge0 = agen.agency_trip_rate
            agen.between_weight_ge0 = agen.agency_trip_rate
            agen.base_ge0_equity_metrics = \
                tf.reduce_sum(
                    agen.base_entropy_tensor * agen.agent_zone_group_trip_ratio_tensor)
            agen.between_agency_ge0_equity_metrics = \
                tf.math.log(self.mean_accessibility / agen.mean_accessibility)
            print("agency:", agen.agency_id, " base_ge0_equity_metrics:", agen.base_ge0_equity_metrics)
            print("agency:", agen.agency_id,
                  " between_agency_ge0_equity_metrics:", agen.between_agency_ge0_equity_metrics)
            print("agency:", agen.agency_id,
                  " total", agen.base_ge0_equity_metrics * agen.within_weight_ge0 +
                  agen.between_agency_ge0_equity_metrics * agen.between_weight_ge0)
            params[str('agency_') + str(agen.agency_id) + '_base_ge0_equity_index'] = \
                agen.base_ge0_equity_metrics.numpy()
            params[str('agency_') + str(agen.agency_id) + '_between_agency_ge0_equity_index'] = \
                agen.between_agency_ge0_equity_metrics.numpy()
            params[str('agency_') + str(agen.agency_id) + '_within_weight_ge0'] = agen.within_weight_ge0.numpy()
            params[str('agency_') + str(agen.agency_id) + '_between_weight_ge0'] = agen.between_weight_ge0.numpy()
        print("base_ge0_equity_metrics:", self.base_ge0_equity_metrics)
        self.base_equity_metrics = self.base_ge0_equity_metrics
        params['equity_index'] = self.base_ge0_equity_metrics.numpy()

    def calculate_base_ge1_equity_index(self):
        # calculate the base theil index
        self.base_entropy_tensor2 = tf.math.log(self.base_zone_transit_accessibility / self.mean_accessibility) * \
                                    (self.base_zone_transit_accessibility / self.mean_accessibility)
        self.base_ge1_entropy_metrics = tf.reduce_sum(self.base_entropy_tensor2 * self.zone_group_trip_ratio_tensor)
        for agen in g_agency_list:
            agen.base_entropy_tensor2 = tf.math.log(agen.base_zone_transit_accessibility / agen.mean_accessibility) * \
                                        (agen.base_zone_transit_accessibility / agen.mean_accessibility)
            agen.within_weight_ge1 = agen.agency_trip_rate * (agen.mean_accessibility / self.mean_accessibility)
            agen.between_weight_ge1 = agen.agency_trip_rate
            agen.base_ge1_entropy_metrics = \
                tf.reduce_sum(
                    agen.base_entropy_tensor2 * agen.agent_zone_group_trip_ratio_tensor)
            agen.between_agency_ge1_entropy_metrics = \
                tf.math.log(agen.mean_accessibility / self.mean_accessibility) * \
                (agen.mean_accessibility / self.mean_accessibility)
            print("agency:", agen.agency_id, " base_ge1_entropy_metrics:", agen.base_ge1_entropy_metrics)
            print("agency:", agen.agency_id,
                  " between_agency_ge1_entropy_metrics:", agen.between_agency_ge1_entropy_metrics)
            print("agency:", agen.agency_id,
                  " total", agen.base_ge1_entropy_metrics * agen.within_weight_ge1 +
                  agen.between_agency_ge1_entropy_metrics * agen.between_weight_ge1)
            params[str('agency_') + str(agen.agency_id) + '_base_ge1_equity_index'] = \
                agen.base_ge1_entropy_metrics.numpy()
            params[str('agency_') + str(agen.agency_id) + '_between_agency_ge1_equity_index'] = \
                agen.between_agency_ge1_entropy_metrics.numpy()
            params[str('agency_') + str(agen.agency_id) + '_within_weight_ge1'] = agen.within_weight_ge1.numpy()
            params[str('agency_') + str(agen.agency_id) + '_between_weight_ge1'] = agen.between_weight_ge1.numpy()

        print("base_ge1_entropy_metrics:", self.base_ge1_entropy_metrics)
        self.base_equity_metrics = self.base_ge1_entropy_metrics
        params['equity_index'] = self.base_ge1_entropy_metrics.numpy()

    def calculate_base_cv(self):
        # calculate the coefficient of variation
        self.base_cv = tf.math.reduce_std(self.base_zone_transit_accessibility) / tf.math.reduce_mean(
            self.base_zone_transit_accessibility)

        print("base_cv:", self.base_cv)
        self.base_equity_metrics = self.base_cv
        params['equity_index'] = self.base_cv.numpy()

    def calculate_base_gini_coefficient(self):
        # calculate the gini coefficient
        self.base_gini_coefficient = 0
        for index1 in self.zone_group_id_list:
            for index2 in self.zone_group_id_list:
                self.base_gini_coefficient = self.base_gini_coefficient + \
                                             tf.math.abs(self.base_zone_transit_accessibility[:, index1] -
                                                         self.base_zone_transit_accessibility[:, index2]) * \
                                             self.zone_group_trip_ratio_tensor[:, index1] * \
                                             self.zone_group_trip_ratio_tensor[:, index2]
        self.base_gini_coefficient = self.base_gini_coefficient / (2 * self.mean_accessibility)
        print("base_gini_coefficient:", self.base_gini_coefficient)
        self.base_equity_metrics = self.base_gini_coefficient
        params['equity_index'] = self.base_gini_coefficient.numpy()

    def calculate_base_atkinson_index(self, epsilon=0.5):
        # calculate the atkinson index
        self.base_av_mean = self.zone_group_trip_ratio_tensor * \
                            self.base_zone_transit_accessibility ** (1 - epsilon)
        self.base_atkinson_index = ((tf.reduce_sum(self.base_av_mean)) ** (1 / (1 - epsilon))) / self.mean_accessibility
        self.base_atkinson_index = 1 - self.base_atkinson_index
        print("base_atkinson_index:", self.base_atkinson_index)
        self.base_equity_metrics = self.base_atkinson_index
        params['equity_index'] = self.base_atkinson_index.numpy()

    def calculate_base_revenue(self):
        self.od_group_base_income_tensor = \
            self.od_group_trip_tensor * self.base_transit_prob * self.transit_travel_cost_tensor
        self.base_income = tf.reduce_sum(self.od_group_base_income_tensor)
        self.od_group_base_subsidy_cost_tensor = \
            self.od_group_trip_tensor * self.base_transit_prob * self.od_group_initial_subsidy_tensor
        self.base_subsidy_cost = tf.reduce_sum(self.od_group_base_subsidy_cost_tensor)
        self.od_group_base_revenue_tensor = self.od_group_base_income_tensor - self.od_group_base_subsidy_cost_tensor
        self.base_revenue = tf.reduce_sum(self.od_group_base_revenue_tensor)
        for agen in g_agency_list:
            start, end = self.obtain_od_group_agency_range(agen.agency_id)
            agen.base_income = tf.reduce_sum(self.od_group_base_income_tensor[:, start:end])
            agen.base_subsidy_cost = tf.reduce_sum(self.od_group_base_subsidy_cost_tensor[:, start:end])
            agen.base_revenue = tf.reduce_sum(self.od_group_base_revenue_tensor[:, start:end])
            print("agency:", agen.agency_id, " base_income:", agen.base_income)
            print("agency:", agen.agency_id, " base_subsidy_cost:", agen.base_subsidy_cost)
            print("agency:", agen.agency_id, " base_revenue:", agen.base_revenue)
            params[str('agency_') + str(agen.agency_id) + '_base_income'] = agen.base_income.numpy()
            params[str('agency_') + str(agen.agency_id) + '_base_subsidy_cost'] = agen.base_subsidy_cost.numpy()
            params[str('agency_') + str(agen.agency_id) + '_base_transit_revenue'] = agen.base_revenue.numpy()

        print("base_income:", self.base_income.numpy())
        print("base_subsidy_cost:", self.base_subsidy_cost.numpy())
        print("base_revenue:", self.base_revenue.numpy())
        params['base_income'] = self.base_income.numpy()
        params['base_subsidy_cost'] = self.base_subsidy_cost.numpy()
        params['base_revenue'] = self.base_revenue.numpy()

    def create_variables(self):
        # create the variables for the transit subsidy
        self.transit_subsidy_tensor = tf.Variable(self.od_group_initial_subsidy_tensor, dtype=tf.float32,
                                                  constraint=lambda x: tf.clip_by_value(x, 0, 10))


# ==================== Data Input ====================
def data_input(inp_path):
    # ==================== Read the input data ====================
    # ------------read the settings.csv file--------------
    # part 1: read the pop_group
    print("start to read the settings.csv file...")
    pop_df = read_setting('pop_group', inp_path)
    pop_df.reindex()
    seq_group_dict = dict(zip(pop_df.index, pop_df['group_id']))
    seq_income_dict = dict(zip(pop_df.index, pop_df['annual_income']))
    seq_hourly_pay_dict = dict(zip(pop_df.index, pop_df['hourly_salary']))
    seq_vot_dict = dict(zip(pop_df.index, pop_df['value_of_time']))
    # part 2: read the agency
    agency_df = read_setting('agency', inp_path)
    for index, row in agency_df.iterrows():
        agency_id = int(row['agency_id'])
        agency_name = str(row['agency_name'])
        agen = Agency(agency_id, agency_name)
        g_agency_dict[agency_id] = agen
        g_agency_list.append(agen)
    params['nb_agency'] = len(g_agency_list)

    # ------------read zone.csv file-------------
    print("start to read zone.csv file...")
    with open(inp_path + 'zone.csv', mode='r') as file:
        csv_reader = csv.DictReader(file)
        zone_seq = 0
        for row in csv_reader:
            zone_id = int(row["zone_id"])
            x = float(row["x_coord"])
            y = float(row["y_coord"])
            geometry = row["geometry"]
            pop = float(row["pop"])  # the population living in the zone
            zone_group_list = []
            zone_group_pop_ratio_dict = {}
            zone_group_pop_dict = {}
            zone_group_income_dict = {}
            zone_group_hourly_pay_dict = {}
            zone_group_vot_dict = {}
            zone_agency_id = str(row["agency_id"])
            for g_seq in seq_group_dict.keys():
                grp_id = int(seq_group_dict[g_seq])
                col_name = 'pop_ratio_' + str(grp_id)
                zone_group_list.append(grp_id)
                zone_group_pop_ratio_dict[grp_id] = float(row[col_name])
                zone_group_pop_dict[grp_id] = pop * zone_group_pop_ratio_dict[grp_id]
                zone_group_income_dict[grp_id] = float(seq_income_dict[g_seq])
                zone_group_hourly_pay_dict[grp_id] = float(seq_hourly_pay_dict[g_seq])
                zone_group_vot_dict[grp_id] = float(seq_vot_dict[g_seq])

            # instantiate the zone object
            zone = Zone(zone_id, zone_seq, x, y, geometry, pop, zone_group_list, zone_group_pop_dict,
                        zone_group_pop_ratio_dict, zone_group_income_dict, zone_group_hourly_pay_dict,
                        zone_group_vot_dict, zone_agency_id)

            g_zone_list.append(zone)
            g_zone_id_seq_dict[zone_id] = zone_seq
            g_zone_seq_id_dict[zone_seq] = zone_id
            g_agency_dict[int(zone_agency_id)].zone_list.append(zone)
            g_zone_id_agency_id_dict[zone_id] = int(zone_agency_id)

            zone_seq += 1
            if len(g_zone_list) % 1 == 0:
                print('read ' + str(len(g_zone_list)) + ' zones...')

    # ------------calculate basic information for each agency--------------

    # ------------read the demand.csv file--------------
    print("start to read the demand.csv file...")
    with open(inp_path + 'demand.csv', mode='r') as file:
        csv_reader = csv.DictReader(file)
        od_group_seq = 0
        od_seq = 0
        for row in csv_reader:
            demand_id = int(row["demand_id"])
            from_zone_id = int(row["from_zone_id"])
            to_zone_id = int(row["to_zone_id"])
            od_trips = float(row["trips"])
            od_auto_tc = float(row["travel_cost_auto"])
            od_auto_tt = float(row["travel_time_auto"])
            od_transit_tc = float(row["travel_cost_transit"])
            od_transit_tt = float(row["travel_time_transit"])
            agency_id = g_zone_id_agency_id_dict[from_zone_id]
            od_group_dict = {}
            od_group_list = []
            for grp_id in g_zone_list[g_zone_id_seq_dict[from_zone_id]].zone_group_list:
                # step 1: for each od group, calculate the trips
                # step 2: for each od group, calculate the value of time
                od_group_id = demand_id * 1000 + grp_id
                od_group_pop_ratio = g_zone_list[g_zone_id_seq_dict[from_zone_id]].zone_pop_ratio_dict[grp_id]
                od_group_trips = od_trips * od_group_pop_ratio
                od_group_auto_tt = od_auto_tt
                od_group_auto_tc = od_auto_tc
                od_group_transit_tt = od_transit_tt
                od_group_transit_tc = od_transit_tc
                od_group_hourly_pay = g_zone_list[g_zone_id_seq_dict[from_zone_id]].zone_hourly_pay_dict[grp_id]
                od_group_vot = g_zone_list[g_zone_id_seq_dict[from_zone_id]].zone_group_vot_dict[grp_id]

                # instantiate the ODGroup object
                od_group = ODGroup(od_group_id, od_group_seq, demand_id, grp_id, from_zone_id, to_zone_id, od_trips,
                                   od_group_pop_ratio, od_group_trips, od_group_auto_tc,
                                   od_group_auto_tt, od_group_transit_tc, od_group_transit_tt,
                                   od_group_hourly_pay, od_group_vot, agency_id)
                od_group_dict[grp_id] = od_group
                od_group_list.append(od_group)
                od_group_seq += 1
            # instantiate the demand object
            demand = Demand(demand_id, od_seq, from_zone_id, to_zone_id, od_trips, od_group_dict,
                            od_group_list, g_zone_list[g_zone_id_seq_dict[from_zone_id]], agency_id)
            g_demand_list.append(demand)
            g_zone_list[g_zone_id_seq_dict[from_zone_id]].od_list.append(demand)
            g_agency_dict[agency_id].od_list.append(demand)
            od_seq += 1
            if len(g_demand_list) % 1 == 0:
                print('read ' + str(len(g_demand_list)) + ' od demands...')

    print("start to calculate basic information...")
    params['total_pop'], params['mean_vot'] = calculate_mean_vot(g_zone_list)
    params['nb_zones'] = len(g_zone_list)
    params['nb_zone_groups'] = len(g_zone_list) * len(seq_group_dict.keys())
    params['nb_od_groups'] = od_group_seq
    params['total_trips'] = 0
    for od in g_demand_list:
        params['total_trips'] += od.trips
    print("total_pop: ", params['total_pop'])
    print("mean_vot: ", params['mean_vot'])
    print("nb_zones: ", params['nb_zones'])
    print("nb_zone_groups: ", params['nb_zone_groups'])
    print("nb_od_groups: ", params['nb_od_groups'])
    print("total_trips: ", params['total_trips'])

    # calculate the total trips for each agency
    for agen in g_agency_list:
        agen.calculate_agency_total_population()
        agen.calculate_agency_group_population()
        agen.calculate_agency_mean_vot()
        agen.calculate_agency_total_trips()

    for zone_obj in g_zone_list:
        zone_obj.zone_total_trips = 0
        for od in zone_obj.od_list:
            zone_obj.zone_total_trips += od.trips
        zone_obj.trip_ratio = zone_obj.zone_total_trips / params['total_trips']  # the ratio of the trips in the zone
        zone_obj.agency_trip_ratio = zone_obj.zone_total_trips / g_agency_dict[int(zone_obj.agency_id)].total_trips
        # the ratio of the trips in the agency
        print("zone:", zone_obj.zone_id, " total_trips:", zone_obj.zone_total_trips,
              " trip_ratio:", zone_obj.trip_ratio,
              " agency_trip_ratio:", zone_obj.agency_trip_ratio)

    print("data input finished!")


def computationalGraph(inp_path):
    # calculate the length of the tensor
    od_group_id = 0
    zone_group_id = 0
    od_group_id_list = []
    zone_id_list = []
    zone_group_id_list = []
    inner_zone_group_id_list = []
    zone_agency_id_list = []
    od_agency_id_list = []
    demand_id_list = []
    from_zone_id_list = []
    to_zone_id_list = []
    od_group_id_list = []
    inner_od_group_id_list = []
    od_group_trip_list = []
    zone_trip_ratio_list = []
    zone_trip_ratio_agency_list = []
    zone_pop_list = []
    auto_travel_cost_list = []
    auto_travel_time_list = []
    transit_travel_cost_list = []
    transit_travel_time_list = []
    vot_list = []
    hourly_salary_list = []
    zone_pop_ratio_list = []
    od_group_initial_subsidy_list = []
    for agen in g_agency_list:
        for zone_obj in agen.zone_list:
            for zone_group in list(zone_obj.zone_group_dict.values()):
                zone_group_id_list.append(zone_group_id)  # 1
                zone_id_list.append(zone_obj.zone_id)  # 2
                inner_zone_group_id_list.append(zone_group.group_id)  # 3
                zone_agency_id_list.append(agen.agency_id)  # 4
                zone_trip_ratio_list.append(zone_obj.trip_ratio)  # 5
                zone_trip_ratio_agency_list.append(zone_obj.agency_trip_ratio)  # 6
                zone_pop_ratio_list.append(zone_group.pop_ratio)  # 7
                zone_pop_list.append(zone_group.pop)  # 8
                for od in zone_obj.od_list:
                    for od_group in od.od_group_list:
                        if (od_group.group_id == zone_group.group_id) & (od_group.from_zone_id == zone_obj.zone_id):
                            od_group_id_list.append(od_group_id)  # 9
                            od_agency_id_list.append(agen.agency_id)  # 10
                            demand_id_list.append(od_group.demand_id)  # 11
                            from_zone_id_list.append(od_group.from_zone_id)  # 12
                            to_zone_id_list.append(od_group.to_zone_id)  # 13
                            inner_od_group_id_list.append(od_group.group_id)  # 14
                            od_group_trip_list.append(od_group.group_trips)  # 15
                            auto_travel_cost_list.append(od_group.auto_tc)  # 16
                            auto_travel_time_list.append(od_group.auto_tt)  # 17
                            transit_travel_cost_list.append(od_group.transit_tc)  # 18
                            transit_travel_time_list.append(od_group.transit_tt)  # 19
                            vot_list.append(od_group.od_group_vot)  # 20
                            hourly_salary_list.append(od_group.od_group_hourly_pay)  # 21
                            od_group_initial_subsidy_list.append(od_group.initial_subsidy)  # 22
                            g_od_group_zone_group_dict[od_group_id] = zone_group_id
                            od_group_id += 1
                zone_group_id += 1
    nb_zone_groups = zone_group_id
    nb_od_groups = od_group_id
    # generate an incidence matrix for according to g_od_group_zone_group_dict

    incidence_matrix = np.zeros((nb_od_groups, nb_zone_groups))
    for od_group_id in g_od_group_zone_group_dict.keys():
        incidence_matrix[od_group_id, g_od_group_zone_group_dict[od_group_id]] = 1
    # print("incidence_matrix:", incidence_matrix)
    # generate a csv for incidence matrix
    with open(inp_path + 'incidence_matrix.csv', mode='w', newline='') as file:
        wrt = csv.writer(file)
        wrt.writerows(incidence_matrix)

    graph = ComputationalGraph(zone_group_id_list, zone_id_list,
                               inner_zone_group_id_list, zone_agency_id_list,
                               zone_trip_ratio_list, zone_trip_ratio_agency_list,
                               zone_pop_ratio_list, zone_pop_list,
                               od_group_id_list, od_agency_id_list,
                               demand_id_list, from_zone_id_list,
                               to_zone_id_list, inner_od_group_id_list,
                               od_group_trip_list, auto_travel_cost_list,
                               auto_travel_time_list, transit_travel_cost_list,
                               transit_travel_time_list, vot_list,
                               hourly_salary_list, incidence_matrix, od_group_initial_subsidy_list)

    # generate two csv files for graph including all attributes of class ComputationalGraph
    # first file include all attributes of od group
    # write in the csv for all attributes in graph
    if log_flag:
        with open(inp_path + 'base_od_group_graph.csv', mode='w', newline='') as file:
            fieldnames = ['od_group_id', 'agency_id', 'demand_id', 'from_zone_id', 'to_zone_id', 'group_id',
                          'od_group_trips', 'od_group_trips_ratio', 'od_group_trips_ratio_in_zone',
                          'auto_travel_cost', 'auto_travel_time',
                          'transit_travel_cost', 'transit_travel_time', 'od_group_hourly_pay', 'od_group_vot',
                          'base_auto_utility', 'base_transit_utility', 'base_auto_prob', 'base_transit_prob',
                          'base_od_transit_access', 'base_od_group_income', 'base_od_group_subsidy_cost',
                          'base_od_group_revenue']
            wrt = csv.DictWriter(file, fieldnames=fieldnames)
            wrt.writeheader()
            for od_group_id in range(len(graph.od_group_id_list)):
                wrt.writerow({'od_group_id': graph.od_group_id_list[od_group_id],
                              'agency_id': graph.od_agency_id_list[od_group_id],
                              'demand_id': graph.demand_id_list[od_group_id],
                              'from_zone_id': graph.from_zone_id_list[od_group_id],
                              'to_zone_id': graph.to_zone_id_list[od_group_id],
                              'group_id': graph.inner_od_group_id_list[od_group_id],
                              'od_group_trips': graph.od_group_trip_tensor[0, od_group_id].numpy(),
                              'od_group_trips_ratio': graph.od_group_trip_ratio_tensor[0, od_group_id].numpy(),
                              'od_group_trips_ratio_in_zone': graph.od_trip_ratio_in_zone[0, od_group_id].numpy(),
                              'auto_travel_cost': graph.auto_travel_cost_tensor[0, od_group_id].numpy(),
                              'auto_travel_time': graph.auto_travel_time_tensor[0, od_group_id].numpy(),
                              'transit_travel_cost': graph.transit_travel_cost_tensor[0, od_group_id].numpy(),
                              'transit_travel_time': graph.transit_travel_time_tensor[0, od_group_id].numpy(),
                              'od_group_hourly_pay': graph.hourly_salary_tensor[0, od_group_id].numpy(),
                              'od_group_vot': graph.vot_tensor[0, od_group_id].numpy(),
                              'base_auto_utility': graph.base_auto_utility[0, od_group_id].numpy(),
                              'base_transit_utility': graph.base_transit_utility[0, od_group_id].numpy(),
                              'base_auto_prob': graph.base_auto_prob[0, od_group_id].numpy(),
                              'base_transit_prob': graph.base_transit_prob[0, od_group_id].numpy(),
                              'base_od_transit_access': graph.base_od_transit_accessibility[0, od_group_id].numpy(),
                              'base_od_group_income': graph.od_group_base_income_tensor[0, od_group_id].numpy(),
                              'base_od_group_subsidy_cost': graph.od_group_base_subsidy_cost_tensor[
                                  0, od_group_id].numpy(),
                              'base_od_group_revenue': graph.od_group_base_revenue_tensor[0, od_group_id].numpy()})

        with open(inp_path + 'base_zone_group_graph.csv', mode='w', newline='') as file:
            fieldnames = ['zone_group_id', 'zone_id', 'agency_id', 'group_id', 'zone_group_trip',
                          'zone_group_trip_ratio',
                          'zone_trip_ratio', 'zone_trip_ratio_agency',
                          'zone_pop_ratio', 'zone_pop', 'base_zone_transit_accessibility']
            wrt = csv.DictWriter(file, fieldnames=fieldnames)
            wrt.writeheader()
            for zone_group_id in range(len(graph.zone_group_id_list)):
                wrt.writerow({'zone_group_id': graph.zone_group_id_list[zone_group_id],
                              'zone_id': graph.zone_id_list[zone_group_id],
                              'agency_id': graph.zone_agency_id_list[zone_group_id],
                              'group_id': graph.inner_zone_group_id_list[zone_group_id],
                              'zone_group_trip': graph.zone_group_trips_tensor[0, zone_group_id].numpy(),
                              'zone_group_trip_ratio': graph.zone_group_trip_ratio_tensor[0, zone_group_id].numpy(),
                              'zone_trip_ratio': graph.zone_trip_ratio_tensor[0, zone_group_id].numpy(),
                              'zone_trip_ratio_agency': graph.zone_trip_ratio_agency_tensor[0, zone_group_id].numpy(),
                              'zone_pop_ratio': graph.zone_pop_ratio_tensor[0, zone_group_id].numpy(),
                              'zone_pop': graph.zone_pop_tensor[0, zone_group_id].numpy(),
                              'base_zone_transit_accessibility':
                                  graph.base_zone_transit_accessibility[0, zone_group_id].numpy()})

        with open(inp_path + 'base_params.csv', mode='w', newline='') as file:
            fieldnames = ['params', 'value']
            wrt = csv.DictWriter(file, fieldnames=fieldnames)
            wrt.writeheader()
            for key in params.keys():
                wrt.writerow({'params': key, 'value': params[key]})

    print("finish writing the od_group_graph.csv and zone_group_graph.csv files...")
    return graph


# ==================== Calculate functions ====================
def read_setting(parameter_name, inp_path):
    j_parameter_name = '[' + parameter_name + ']'
    attribute_list = []
    parameter_flag = ''
    nb_columns = 0
    columns_name = []
    with open(inp_path + 'settings.csv', 'r') as file:
        # know the name of the columns
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # remove items in list with ''
            if row[0] != '':
                if row[0] == j_parameter_name:
                    # remove the '' in the list
                    row = list(filter(None, row))
                    parameter_flag = j_parameter_name
                    nb_columns = len(row) - 1
                    columns_name = row[1:len(row)]
                else:
                    parameter_flag = ''
                # go to next for loop
                continue
            if parameter_flag == j_parameter_name:
                row = list(filter(None, row))
                if len(row) != 0:
                    attribute_list.append(row[0:nb_columns])  # only insert the non-empty row

    # create a dataframe
    att_df = pd.DataFrame(attribute_list, columns=columns_name)
    # remove the empty rows without any value
    att_df = att_df.dropna(how='all')
    if len(att_df) == 0:
        print('No attribute ' + parameter_name + ' are found in the setting.csv file')
        return None

    return att_df


def calculate_mean_vot(global_zone_list):
    # calculate the total pop
    total_pop = 0
    for zone_obj in global_zone_list:
        total_pop += zone_obj.pop
    # calculate the percentage of population for each zone
    mean_vot = 0
    for zone_obj in global_zone_list:
        zone_obj.pop_ratio = zone_obj.pop / total_pop
        # calculate pop_ratio_weighted_vot
        # pop_ratio * pop_ratio_of_group * hourly_pay
        for grp_id in zone_obj.zone_group_list:
            mean_vot += zone_obj.pop_ratio * zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.zone_group_vot_dict[grp_id]
    return total_pop, mean_vot


def calculate_adjusted_prob(cg):
    cg.adjusted_auto_utility = - cg.auto_travel_time_tensor * cg.vot_tensor - cg.auto_travel_cost_tensor
    cg.adjusted_transit_utility = - cg.transit_travel_time_tensor * cg.vot_tensor \
                                  - cg.transit_travel_cost_tensor + cg.transit_subsidy_tensor
    cg.adjusted_auto_odd = tf.math.exp(cg.adjusted_auto_utility)
    cg.adjusted_transit_odd = tf.math.exp(cg.adjusted_transit_utility)

    # calculate the adjusted probability
    cg.adjusted_auto_prob = cg.adjusted_auto_odd / (cg.adjusted_auto_odd + cg.adjusted_transit_odd)
    cg.adjusted_transit_prob = cg.adjusted_transit_odd / (cg.adjusted_auto_odd + cg.adjusted_transit_odd)
    cg.adjusted_auto_volume = cg.adjusted_auto_prob * cg.od_group_trip_tensor
    cg.adjusted_transit_volume = cg.adjusted_transit_prob * cg.od_group_trip_tensor
    return cg


def calculate_adjusted_accessibility(cg):
    # calculate the adjusted accessibility
    cg.adjusted_transit_utility_mean_vot = \
        - cg.transit_travel_time_tensor * params['mean_vot'] - cg.transit_travel_cost_tensor + cg.transit_subsidy_tensor
    cg.adjusted_od_transit_accessibility = (
        tf.math.maximum(1e-5, 1 - (-cg.adjusted_transit_utility_mean_vot) / cg.hourly_salary_tensor))
    cg.adjusted_od_transit_accessibility = cg.adjusted_od_transit_accessibility * cg.od_trip_ratio_in_zone
    cg.adjusted_zone_transit_accessibility = tf.matmul(cg.adjusted_od_transit_accessibility, cg.incidence_matrix)
    cg.adjusted_trip_weighted_accessibility = cg.adjusted_zone_transit_accessibility * cg.zone_group_trip_ratio_tensor
    cg.adjusted_mean_accessibility = tf.reduce_sum(cg.adjusted_trip_weighted_accessibility)
    return cg


def calculate_adjusted_ge0_equity_index(cg):
    # calculate the base entropy
    cg.adjusted_entropy_tensor = tf.math.log(cg.adjusted_mean_accessibility / cg.adjusted_zone_transit_accessibility)
    cg.adjusted_ge0_equity_metrics = tf.reduce_sum(cg.adjusted_entropy_tensor * cg.zone_group_trip_ratio_tensor)
    cg.adjusted_equity_metrics = cg.adjusted_ge0_equity_metrics
    return cg


def calculate_adjusted_revenue(cg, agen):
    cg.adjusted_transit_volume = cg.adjusted_transit_prob * cg.od_group_trip_tensor
    cg.adjusted_od_group_income_tensor = cg.adjusted_transit_volume * cg.transit_travel_cost_tensor
    cg.adjusted_od_group_subsidy_cost_tensor = cg.adjusted_transit_volume * cg.transit_subsidy_tensor
    cg.adjusted_od_group_revenue_tensor = cg.adjusted_od_group_income_tensor - cg.adjusted_od_group_subsidy_cost_tensor
    # calculate the transit volume
    agen_id = agen.agency_id
    start_odg, end_odg = cg.obtain_od_group_agency_range(agen_id)
    agen.adjusted_income = tf.reduce_sum(cg.adjusted_od_group_income_tensor[0, start_odg:end_odg])
    agen.adjusted_subsidy_cost = tf.reduce_sum(cg.adjusted_od_group_subsidy_cost_tensor[0, start_odg:end_odg])
    agen.adjusted_revenue = tf.reduce_sum(cg.adjusted_od_group_revenue_tensor[0, start_odg:end_odg])
    cg.equity_gap = cg.base_equity_metrics * params['target'] - cg.adjusted_equity_metrics
    cg.equity_improvement = cg.base_equity_metrics - cg.adjusted_equity_metrics
    cg.TEC_demand = cg.equity_gap * 100000
    cg.equity_income = cg.equity_gap * params['TEC_price'] * 100000
    # print("check:", cg.adjusted_od_group_subsidy_cost_tensor)
    total_income = agen.adjusted_income
    total_subsidy = agen.adjusted_subsidy_cost
    equity_income = cg.equity_income
    total_revenue = agen.adjusted_revenue + cg.equity_income
    cg.adjusted_total_revenue = total_revenue
    return total_income, total_subsidy, equity_income, total_revenue


def write_in_class(cg, agen, temp_params):
    # write in the class
    cg.adjusted_auto_utility = temp_params['cg.adjusted_auto_utility']
    cg.adjusted_transit_utility = temp_params['cg.adjusted_transit_utility']
    cg.adjusted_auto_odd = temp_params['cg.adjusted_auto_odd']
    cg.adjusted_transit_odd = temp_params['cg.adjusted_transit_odd']
    cg.adjusted_auto_prob = temp_params['cg.adjusted_auto_prob']
    cg.adjusted_transit_prob = temp_params['cg.adjusted_transit_prob']
    cg.adjusted_auto_volume = temp_params['cg.adjusted_auto_volume']
    cg.adjusted_transit_volume = temp_params['cg.adjusted_transit_volume']
    cg.adjusted_od_group_income_tensor = temp_params['cg.adjusted_od_group_income_tensor']
    cg.adjusted_od_group_subsidy_cost_tensor = temp_params['cg.adjusted_od_group_subsidy_cost_tensor']
    cg.adjusted_od_group_revenue_tensor = temp_params['cg.adjusted_od_group_revenue_tensor']
    cg.adjusted_entropy_tensor = temp_params['cg.adjusted_entropy_tensor']
    cg.adjusted_equity_metrics = temp_params['cg.adjusted_equity_metrics']
    cg.adjusted_mean_accessibility = temp_params['cg.adjusted_mean_accessibility']
    cg.equity_gap = temp_params['cg.equity_gap']
    cg.equity_improvement = temp_params['cg.equity_improvement']
    cg.TEC_demand = temp_params['cg.TEC_demand']
    cg.equity_income = temp_params['cg.equity_income']
    cg.adjusted_total_revenue = temp_params['cg.adjusted_total_revenue']
    cg.adjusted_transit_utility_mean_vot = temp_params['cg.adjusted_transit_utility_mean_vot']
    cg.adjusted_od_transit_accessibility = temp_params['cg.adjusted_od_transit_accessibility']
    cg.adjusted_zone_transit_accessibility = temp_params['cg.adjusted_zone_transit_accessibility']
    cg.adjusted_trip_weighted_accessibility = temp_params['cg.adjusted_trip_weighted_accessibility']
    cg.adjusted_mean_accessibility = temp_params['cg.adjusted_mean_accessibility']
    agen.adjusted_income = temp_params['agen.adjusted_income']
    agen.adjusted_subsidy_cost = temp_params['agen.adjusted_subsidy_cost']
    agen.adjusted_revenue = temp_params['agen.adjusted_revenue']
    return cg, agen


# speed up using tf.function
# @tf.function
def est_gradient(cg, agen, opt, nb_epoch, method):
    # calculate the gradients
    un_mask = cg.obtain_unmask_index(agen.agency_id)
    temp_params = {}
    with tf.GradientTape(persistent=True) as tape:
        cg = calculate_adjusted_prob(cg)
        cg = calculate_adjusted_accessibility(cg)
        cg = calculate_adjusted_ge0_equity_index(cg)
        t_income, t_subsidy, t_equity_income, t_revenue = calculate_adjusted_revenue(cg, agen)
        loss = - t_revenue
        loss_income = - t_income + t_subsidy
        loss_equity = - t_equity_income
        # calculate the gradients
    # print(cg.transit_subsidy_tensor)
    grad = tape.gradient(loss, cg.transit_subsidy_tensor)
    # grad_income = tape.gradient(loss_income, cg.transit_subsidy_tensor)
    # grad_equity = tape.gradient(loss_equity, cg.transit_subsidy_tensor)
    un_mask = tf.cast(un_mask, dtype=tf.float32)
    grad = grad * un_mask
    temp_params['cg.adjusted_auto_utility'] = cg.adjusted_auto_utility
    temp_params['cg.adjusted_transit_utility'] = cg.adjusted_transit_utility
    temp_params['cg.adjusted_auto_odd'] = cg.adjusted_auto_odd
    temp_params['cg.adjusted_transit_odd'] = cg.adjusted_transit_odd
    temp_params['cg.adjusted_auto_prob'] = cg.adjusted_auto_prob
    temp_params['cg.adjusted_transit_prob'] = cg.adjusted_transit_prob
    temp_params['cg.adjusted_auto_volume'] = cg.adjusted_auto_volume
    temp_params['cg.adjusted_transit_volume'] = cg.adjusted_transit_volume
    temp_params['cg.adjusted_od_group_income_tensor'] = cg.adjusted_od_group_income_tensor
    temp_params['cg.adjusted_od_group_subsidy_cost_tensor'] = cg.adjusted_od_group_subsidy_cost_tensor
    temp_params['cg.adjusted_od_group_revenue_tensor'] = cg.adjusted_od_group_revenue_tensor
    temp_params['cg.adjusted_transit_utility_mean_vot'] = cg.adjusted_transit_utility_mean_vot
    temp_params['cg.adjusted_od_transit_accessibility'] = cg.adjusted_od_transit_accessibility
    temp_params['cg.adjusted_zone_transit_accessibility'] = cg.adjusted_zone_transit_accessibility
    temp_params['cg.adjusted_trip_weighted_accessibility'] = cg.adjusted_trip_weighted_accessibility
    temp_params['cg.adjusted_mean_accessibility'] = cg.adjusted_mean_accessibility
    temp_params['cg.adjusted_entropy_tensor'] = cg.adjusted_entropy_tensor
    temp_params['cg.adjusted_equity_metrics'] = cg.adjusted_equity_metrics
    temp_params['cg.adjusted_equity_metrics'] = cg.adjusted_equity_metrics
    temp_params['cg.adjusted_mean_accessibility'] = cg.adjusted_mean_accessibility
    temp_params['cg.equity_gap'] = cg.equity_gap
    temp_params['cg.equity_improvement'] = cg.equity_improvement
    temp_params['cg.TEC_demand'] = cg.TEC_demand
    temp_params['cg.equity_income'] = cg.equity_income
    temp_params['cg.adjusted_total_revenue'] = cg.adjusted_total_revenue
    temp_params['agen.adjusted_income'] = agen.adjusted_income
    temp_params['agen.adjusted_subsidy_cost'] = agen.adjusted_subsidy_cost
    temp_params['agen.adjusted_revenue'] = agen.adjusted_revenue
    return grad, loss, temp_params


# ==================== Main function ====================
if __name__ == '__main__':
    # data_input
    input_path = './'
    TEC_price = 2  # per 0.00001 equity improvement
    target = 0.9
    step_size = 0.0001
    price_not_change_epoch = 5
    params['step_size'] = step_size
    params['TEC_price'] = TEC_price
    params['target'] = float(target)
    params['equity_method'] = equity_method
    # read the input data
    data_input(input_path)
    graph = computationalGraph(input_path)

    total_epoch = 100
    total_iteration = 1000

    loss_value_list = []
    agent_id_list = []
    gap_list = []
    gradients_list = []
    complementary_list = []
    var_information_list = []
    grad_information_list = []
    tec_price_list = []
    agency_information_list = []
    epoch_list = []
    iteration_list = []
    current_tec_demand = 0
    current_price_not_change_epoch = 0
    prev_complementary = 0
    start_time = time.time()
    for epoch in range(total_epoch):
        print("epoch: ", epoch)
        for agency in g_agency_list:
            print("agency_id: ", agency.agency_id)
            for i in range(total_iteration):
                inner_time_start = time.time()
                epoch_list.append(epoch)
                iteration_list.append(i)
                optimizer = Adam(learning_rate=0.01, epsilon=0.001, amsgrad=True)
                gradients, loss_func, info_dict = est_gradient(graph, agency, optimizer, epoch, equity_method)
                optimizer.apply_gradients([(gradients, graph.transit_subsidy_tensor)])
                upper_bound = graph.transit_travel_cost_tensor
                graph.transit_subsidy_tensor.assign(tf.clip_by_value(graph.transit_subsidy_tensor, 0, upper_bound))
                loss_value_list.append(loss_func.numpy())
                agent_id_list.append(agency.agency_id)
                # calculate the mean value of the gradients
                gradients_list.append(tf.reduce_mean(gradients).numpy())
                agency.loss_func = loss_func
                print("loss: ", abs(loss_func.numpy()))
                gap = abs(agency.loss_func - agency.prev_loss) / abs(agency.prev_loss)
                params[str(epoch) + '_agency' + str(agency.agency_id) + '_adjusted_transit_revenue'] = \
                    info_dict['cg.adjusted_total_revenue'].numpy()
                # convert to %
                gap = gap * 100
                print("iteration: ", i, " total_revenue: ",
                      params[str(epoch) + '_agency' + str(agency.agency_id) + '_adjusted_transit_revenue'])
                print("gap: ", gap.numpy(), "%")

                mask = tf.cast(graph.obtain_unmask_index(agency.agency_id), dtype=tf.float32)
                graph.transit_subsidy_tensor * gradients
                # absolte sum
                graph.complementary = tf.reduce_mean(tf.abs(graph.transit_subsidy_tensor * gradients))
                for index in graph.od_group_id_list:
                    grad_information_list.append([graph.from_zone_id_list[index], graph.to_zone_id_list[index],
                                                  graph.od_agency_id_list[index], graph.inner_od_group_id_list[index],
                                                  gradients[0][index].numpy(), graph.complementary.numpy(),
                                                  agency.agency_id, epoch, i])

                avg_complementary = graph.complementary.numpy()
                print("avg_complementary: ", avg_complementary)
                gap_list.append(gap.numpy())
                complementary_list.append(avg_complementary)
                if i == 0:
                    prev_complementary = avg_complementary + 1e-8
                    complementary_gap = 1
                else:
                    complementary_gap = 100 * abs(avg_complementary - prev_complementary) / abs(prev_complementary)
                    prev_complementary = avg_complementary + 1e-8
                print("complementary_gap: ", complementary_gap, '%')
                if (gap < 1e-3) | (complementary_gap < 1e-5):
                    break
                agency.prev_loss = loss_func
                prev_complementary = avg_complementary
                inner_time_end = time.time()
                print("time of iteration: ", inner_time_end - inner_time_start, " seconds")
                # print("gradients: ", gradients)
            for index in graph.od_group_id_list:
                var_information_list.append([graph.from_zone_id_list[index], graph.to_zone_id_list[index],
                                             graph.od_agency_id_list[index], graph.inner_od_group_id_list[index],
                                             graph.transit_subsidy_tensor[0][index].numpy(),
                                             graph.base_od_transit_accessibility[0][index].numpy(),
                                             graph.adjusted_od_transit_accessibility[0][index].numpy(), epoch])
            params[str(epoch) + '_agency' + str(agency.agency_id) + '_total_subsidy'] = \
                info_dict['agen.adjusted_subsidy_cost'].numpy()
            params[str(epoch) + '_agency' + str(agency.agency_id) + '_total_income'] = \
                info_dict['agen.adjusted_income'].numpy()
            params[str(epoch) + '_agency' + str(agency.agency_id) + '_adjusted_equity_index'] = \
                info_dict['cg.adjusted_equity_metrics'].numpy()
            params[str(epoch) + '_agency' + str(agency.agency_id) + '_equity_improvement'] = \
                info_dict['cg.equity_improvement'].numpy()
            params[str(epoch) + '_agency' + str(agency.agency_id) + '_equity_gap'] = \
                info_dict['cg.equity_gap'].numpy()
            params[str(epoch) + '_agency' + str(agency.agency_id) + '_TEC_demand'] = \
                info_dict['cg.TEC_demand'].numpy()
            params[str(epoch) + '_agency' + str(agency.agency_id) + '_equity_income'] = \
                info_dict['cg.equity_income'].numpy()
            params[str(epoch) + '_agency' + str(agency.agency_id) + '_adjusted_total_revenue'] = \
                info_dict['cg.adjusted_total_revenue'].numpy()
            params[str(epoch) + '_agency' + str(agency.agency_id) + '_adjusted_mean_accessibility'] = \
                info_dict['cg.adjusted_mean_accessibility'].numpy()

            print("total_income: ", info_dict['agen.adjusted_income'].numpy())
            print("total_subsidy: ", info_dict['agen.adjusted_subsidy_cost'].numpy())
            print("total_revenue: ", info_dict['agen.adjusted_revenue'].numpy())
            print("base_equity_index: ", graph.base_equity_metrics.numpy())
            print("adjusted_equity_index: ", info_dict['cg.adjusted_equity_metrics'].numpy())
            print("equity_improvement: ", info_dict['cg.equity_improvement'].numpy())
            print("equity_gap: ", info_dict['cg.equity_gap'].numpy())
            print("TEC_demand: ", info_dict['cg.TEC_demand'].numpy())
            print("TEC_price: ", params['TEC_price'])
            print("equity_income: ", info_dict['cg.equity_income'].numpy())
            print("adjusted_total_revenue: ", info_dict['cg.adjusted_total_revenue'].numpy())

            # agency_info = [epoch, agency.agency_id, params['TEC_price'], agency.adjusted_revenue.numpy(),
            #                agency.adjusted_subsidy_cost.numpy(), params['equity_index'],
            #                params[str(epoch) + '_agency' + str(agency.agency_id) + '_adjusted_equity_index'],
            #                graph.adjusted_equity_metrics.numpy(), graph.equity_improvement.numpy(),
            #                graph.equity_gap.numpy(), graph.TEC_demand.numpy()]
            # rewrite agency_info using info_dict
            agency_info = [epoch, agency.agency_id, params['TEC_price'], info_dict['agen.adjusted_revenue'].numpy(),
                           info_dict['agen.adjusted_subsidy_cost'].numpy(), params['equity_index'],
                           info_dict['cg.adjusted_equity_metrics'].numpy(),
                           info_dict['cg.equity_improvement'].numpy(),
                           info_dict['cg.equity_gap'].numpy(), info_dict['cg.TEC_demand'].numpy()]
            current_tec_demand = info_dict['cg.TEC_demand'].numpy()
            write_in_class(graph, agency, info_dict)
            agency_information_list.append(agency_info)

        params['TEC_price'] = (params['TEC_price'] -
                               params['step_size'] *
                               current_tec_demand)
        params['TEC_price'] = max(0.0, params['TEC_price'])
        params['step_size'] = params['step_size'] * 0.99
        print("TEC_price: ", params['TEC_price'])
        # if price does not change for given iterations, then stop
        if abs(np.round(current_tec_demand, 0) * params['TEC_price']) < 0.0001:
            current_price_not_change_epoch += 1
            if current_price_not_change_epoch > price_not_change_epoch:
                break
                # stop the loop epoch
        TEC_price = params['TEC_price']
        end_time = time.time()
        print("total time of epoch: ", end_time - start_time, " seconds")
        print("====================================================================")
    # export the results to csv
    with open(input_path + 'o_output_var.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['from_node_id', 'to_node_id', 'agency_id', 'group_id', 'subsidy', 'base_transit_accessibility',
                         'adjusted_transit_accessibility', 'epoch'])
        for var_info in var_information_list:
            writer.writerow(var_info)

    with open(input_path + 'o_output_grad.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['from_zone_id', 'to_zone_id', 'agency_id', 'group_id', 'gradients', 'complementary', 'agency_id',
             'epoch', 'iteration'])
        for var_info in grad_information_list:
            writer.writerow(var_info)

    # write the accessibility to csv
    with open(input_path + 'o_output_zone_accessibility.csv', mode='w', newline='') as file:
        fieldnames = ['zone_group_id', 'zone_id', 'agency_id', 'group_id', 'zone_group_trip', 'zone_group_trip_ratio',
                      'zone_trip_ratio', 'zone_trip_ratio_agency',
                      'zone_pop_ratio', 'zone_pop', 'base_zone_transit_accessibility',
                      'adjusted_zone_transit_accessibility']
        wrt = csv.DictWriter(file, fieldnames=fieldnames)
        wrt.writeheader()
        for zone_group_id in range(len(graph.zone_group_id_list)):
            wrt.writerow({'zone_group_id': graph.zone_group_id_list[zone_group_id],
                          'zone_id': graph.zone_id_list[zone_group_id],
                          'agency_id': graph.zone_agency_id_list[zone_group_id],
                          'group_id': graph.inner_zone_group_id_list[zone_group_id],
                          'zone_group_trip': graph.zone_group_trips_tensor[0, zone_group_id].numpy(),
                          'zone_group_trip_ratio': graph.zone_group_trip_ratio_tensor[0, zone_group_id].numpy(),
                          'zone_trip_ratio': graph.zone_trip_ratio_tensor[0, zone_group_id].numpy(),
                          'zone_trip_ratio_agency': graph.zone_trip_ratio_agency_tensor[0, zone_group_id].numpy(),
                          'zone_pop_ratio': graph.zone_pop_ratio_tensor[0, zone_group_id].numpy(),
                          'zone_pop': graph.zone_pop_tensor[0, zone_group_id].numpy(),
                          'base_zone_transit_accessibility':
                              graph.base_zone_transit_accessibility[0, zone_group_id].numpy(),
                          'adjusted_zone_transit_accessibility':
                              info_dict['cg.adjusted_zone_transit_accessibility'][0, zone_group_id].numpy()})

    # export dictionary params to csv
    with open(input_path + 'o_output_params.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['param', 'value'])
        for key, value in params.items():
            writer.writerow([key, value])

    # write iteration results to csv
    with open(input_path + 'o_output_iteration.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'iteration', 'loss', 'gap', 'gradients', 'complementary', 'agency_id'])
        for i in range(len(loss_value_list)):
            writer.writerow([epoch_list[i], iteration_list[i], loss_value_list[i], gap_list[i], gradients_list[i],
                             complementary_list[i], agent_id_list[i]])

    end_time = time.time()
    print("total time: ", end_time - start_time)