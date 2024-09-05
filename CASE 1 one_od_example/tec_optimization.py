import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
import csv

# show all columns
baseline_method = 'shared'
# baseline_method = 'internal'
pd.set_option('display.max_columns', None)
ORIG_ATK_PARAMETER = 0.5
ATK_PARAMETER = 1 - ORIG_ATK_PARAMETER
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
log_flag = False


# optimizer = SGD(learning_rate=0.001, momentum=0.9)
# ==================== Class definition ====================
class Agency:
    def __init__(self, agency_id, agency_name):
        self.agency_id = agency_id
        self.agency_name = agency_name
        self.pop = 0
        self.mean_vot = 0
        self.agency_trips = 0
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
            self.agency_trips += od.trips
        print("agency:", self.agency_id, " trips:", self.agency_trips)


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
        for group_id in self.zone_group_list:
            zone_group = ZoneGroup(self.zone_id, group_id, self.zone_pop_dict[group_id],
                                   self.zone_pop_ratio_dict[group_id], self.zone_hourly_pay_dict[group_id],
                                   self.zone_income_dict[group_id], self.zone_group_vot_dict[group_id], self.agency_id)
            self.zone_group_dict[group_id] = zone_group


class ZoneGroup:
    def __init__(self, zone_id, group_id, pop, pop_ratio, hourly_pay, income, vot, agency_id):
        self.zone_id = zone_id
        self.group_id = group_id
        self.zone_group_id = 100 * zone_id + group_id
        self.pop = pop
        self.pop_ratio = pop_ratio
        self.hourly_pay = hourly_pay
        self.income = income
        self.vot = vot
        self.agency_id = agency_id
        self.accessibility = None
        self.pop_weighted_accessibility = None
        self.group_total_trips = None
        self.group_trips_ratio = None
        self.base_entropy = None
        self.theil_index = None
        self.adjusted_accessibility = None
        self.adjusted_pop_weighted_accessibility = None
        self.adjusted_entropy = None
        self.adjusted_theil_index = None
        self.agency_pop_weighted_accessibility = None
        self.adjusted_agency_pop_weighted_accessibility = None
        self.agency_base_entropy = None
        self.agency_theil_index = None
        self.agency_adjusted_entropy = None
        self.agency_adjusted_theil_index = None


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
        self.transit_tc = transit_tc
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

    def calculate_base_utility(self):
        self.base_auto_utility = -self.auto_tc - self.auto_tt * self.od_group_vot
        self.base_transit_utility = -self.transit_tc - self.transit_tt * self.od_group_vot
        print("od:", self.demand_id, " group:", self.group_id,
              " base_auto_utility:", np.round(self.base_auto_utility, 3),
              " base_transit_utility:", np.round(self.base_transit_utility, 3))

    def calculate_base_prob(self):
        odd_auto = np.exp(self.base_auto_utility)
        odd_transit = np.exp(self.base_transit_utility)
        self.base_auto_prob = odd_auto / (odd_auto + odd_transit)
        self.base_transit_prob = odd_transit / (odd_auto + odd_transit)
        print("od:", self.demand_id, " group:", self.group_id,
              " base_auto_prob:", np.round(self.base_auto_prob, 4),
              " base_transit_prob:", np.round(self.base_transit_prob, 4))

    def calculate_base_volume(self):
        # calculate the transit volume
        self.base_auto_volume = self.base_auto_prob * self.group_trips
        self.base_transit_volume = self.base_transit_prob * self.group_trips
        print("od:", self.demand_id, " group:", self.group_id,
              " base_auto_volume:", np.round(self.base_auto_volume, 4),
              " base_transit_volume:", np.round(self.base_transit_volume, 4))

    def calculate_od_base_transit_accessibility(self):
        self.base_od_transit_accessibility = (
            max(1e-5, 1 - (self.transit_tc + self.transit_tt * params['mean_vot']) / self.od_group_hourly_pay))
        print("od:", self.demand_id, " group:", self.group_id,
              " base_od_transit_accessibility:", np.round(self.base_od_transit_accessibility, 4))

    def initialize_variables(self):
        # determine the optimal value of the transit subsidy (only consider total revenue and subsidy cost using
        # bisection method)
        # range of the variable is from [0, 1.5 * transit_tc]
        lower_bound = 0
        upper_bound = 1.5 * self.transit_tc
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
        # self.initial_subsidy = 0.0
        print("od:", self.demand_id, " group:", self.group_id, " initial transit subsidy:", self.initial_subsidy)

    def subsidy(self):
        if self.group_id != 100:
            name = str((self.from_zone_id, self.to_zone_id, self.group_id))
            # set lower bound and upper bound of the variable
            self.transit_subsidy = tf.Variable(self.initial_subsidy, dtype=tf.float32, name=name,
                                               constraint=lambda x: tf.clip_by_value(x, 0, 10))

            g_agency_dict[self.agency_id].var_dict[name] = self.transit_subsidy
            g_agency_dict[self.agency_id].var_upper_bound[name] = 1.0 * self.transit_tc
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
        if log_flag:
            print("od:", self.demand_id, " group:", self.group_id,
                  " adjusted_auto_utility:", self.adjusted_auto_utility,
                  " adjusted_transit_utility:", self.adjusted_transit_utility)

    def calculate_adjusted_probability(self):
        odd_auto = tf.math.exp(self.adjusted_auto_utility)
        odd_transit = tf.math.exp(self.adjusted_transit_utility)
        self.adjusted_auto_prob = odd_auto / (odd_auto + odd_transit)
        self.adjusted_transit_prob = odd_transit / (odd_auto + odd_transit)
        if log_flag:
            print("od:", self.demand_id, " group:", self.group_id,
                  " adjusted_auto_prob:", self.adjusted_auto_prob,
                  " adjusted_transit_prob:", self.adjusted_transit_prob)

    def calculate_adjusted_volume(self):
        self.adjusted_auto_volume = self.adjusted_auto_prob * self.group_trips
        self.adjusted_transit_volume = self.adjusted_transit_prob * self.group_trips
        if log_flag:
            print("od:", self.demand_id, " group:", self.group_id,
                  " adjusted_auto_volume:", self.adjusted_auto_volume,
                  " adjusted_transit_volume:", self.adjusted_transit_volume)

    def calculate_od_adjusted_transit_accessibility(self):
        # self.adjusted_od_transit_accessibility = (
        #     max(1e-5, 1 - (self.transit_tc + self.transit_tt * params['mean_vot'] - self.transit_subsidy)
        #         / self.od_group_hourly_pay))
        self.adjusted_od_transit_accessibility = (
            tf.math.maximum(1e-5, 1 - (self.transit_tc + self.transit_tt * params['mean_vot'] - self.transit_subsidy)
                            / self.od_group_hourly_pay))
        if log_flag:
            print("od:", self.demand_id, " group:", self.group_id,
                  " adjusted_od_transit_accessibility:", self.adjusted_od_transit_accessibility)


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
        agency = Agency(agency_id, agency_name)
        g_agency_dict[agency_id] = agency
        g_agency_list.append(agency)
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
                group_id = int(seq_group_dict[g_seq])
                col_name = 'pop_ratio_' + str(group_id)
                zone_group_list.append(group_id)
                zone_group_pop_ratio_dict[group_id] = float(row[col_name])
                zone_group_pop_dict[group_id] = pop * zone_group_pop_ratio_dict[group_id]
                zone_group_income_dict[group_id] = float(seq_income_dict[g_seq])
                zone_group_hourly_pay_dict[group_id] = float(seq_hourly_pay_dict[g_seq])
                zone_group_vot_dict[group_id] = float(seq_vot_dict[g_seq])

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
    print("start to calculate the basic information for each agency...")
    for agency in g_agency_list:
        agency.calculate_agency_total_population()
        agency.calculate_agency_group_population()
        agency.calculate_agency_mean_vot()
    params['nb_zones'] = len(g_zone_list)
    params['total_pop'], params['mean_vot'] = calculate_mean_vot(g_zone_list)

    # ------------read the demand.csv file--------------
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
            for group_id in g_zone_list[g_zone_id_seq_dict[from_zone_id]].zone_group_list:
                # step 1: for each od group, calculate the trips
                # step 2: for each od group, calculate the value of time
                od_group_id = demand_id * 1000 + group_id
                od_group_pop_ratio = g_zone_list[g_zone_id_seq_dict[from_zone_id]].zone_pop_ratio_dict[group_id]
                od_group_trips = od_trips * od_group_pop_ratio
                od_group_auto_tt = od_auto_tt
                od_group_auto_tc = od_auto_tc
                od_group_transit_tt = od_transit_tt
                od_group_transit_tc = od_transit_tc
                od_group_hourly_pay = g_zone_list[g_zone_id_seq_dict[from_zone_id]].zone_hourly_pay_dict[group_id]
                od_group_vot = g_zone_list[g_zone_id_seq_dict[from_zone_id]].zone_group_vot_dict[group_id]

                # instantiate the ODGroup object
                od_group = ODGroup(od_group_id, od_group_seq, demand_id, group_id, from_zone_id, to_zone_id, od_trips,
                                   od_group_pop_ratio, od_group_trips, od_group_auto_tc,
                                   od_group_auto_tt, od_group_transit_tc, od_group_transit_tt,
                                   od_group_hourly_pay, od_group_vot, agency_id)
                od_group_dict[group_id] = od_group
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

    # calculate the total trips for each agency
    for agency in g_agency_list:
        agency.calculate_agency_total_trips()

    params['nb_od_groups'] = od_group_seq
    params['total_trips'] = 0
    for od in g_demand_list:
        params['total_trips'] += od.trips

    print("total_trips: ", params['total_trips'])
    print("finish reading the input data...")


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


# ==================== Calculate functions ====================
def calculate_mean_vot(g_zone_list):
    # calculate the total pop
    total_pop = 0
    for zone in g_zone_list:
        total_pop += zone.pop
    print("total_pop: ", total_pop)
    # calculate the percentage of population for each zone
    mean_vot = 0
    for zone in g_zone_list:
        zone.pop_ratio = zone.pop / total_pop
        # calculate pop_ratio_weighted_vot
        # pop_ratio * pop_ratio_of_group * hourly_pay
        for group_id in zone.zone_group_list:
            mean_vot += zone.pop_ratio * zone.zone_pop_ratio_dict[group_id] * zone.zone_group_vot_dict[group_id]
    print("mean_vot: ", mean_vot)
    return total_pop, mean_vot


def calculate_base_probability():
    print("start to calculate the base utility and base probability for each od group...")
    for od in g_demand_list:
        for od_group in od.od_group_list:
            od_group.calculate_base_utility()
            od_group.calculate_base_prob()
            od_group.calculate_base_volume()


def calculate_base_accessibility():
    print("start to calculate the base utility and base probability for each od group...")
    for od in g_demand_list:
        for od_group in od.od_group_list:
            od_group.calculate_od_base_transit_accessibility()

    for zone_obj in g_zone_list:
        zone_obj.zone_total_trips = 0
        for od in zone_obj.od_list:
            zone_obj.zone_total_trips += od.trips
        zone_obj.trip_ratio = zone_obj.zone_total_trips / params['total_trips']
        zone_obj.agency_trip_ratio = zone_obj.zone_total_trips / g_agency_dict[int(zone_obj.agency_id)].agency_trips

    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            accessibility = 0
            zone_group_total_trips = 0
            zone_total_trips = 0
            for od in zone_obj.od_list:
                accessibility += od.od_group_dict[grp_id].base_od_transit_accessibility
                zone_group_total_trips += od.od_group_dict[grp_id].group_trips
            zone_obj.zone_group_dict[grp_id].accessibility = accessibility
            zone_obj.zone_group_dict[grp_id].group_total_trips = zone_group_total_trips
            zone_obj.zone_group_dict[grp_id].group_trips_ratio = zone_obj.zone_pop_ratio_dict[grp_id]
            # assume that the trips ratio is the same as the pop ratio
            zone_obj.zone_group_dict[grp_id].pop_weighted_accessibility = (
                    zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.trip_ratio * accessibility)

            zone_obj.zone_group_dict[grp_id].agency_pop_weighted_accessibility = (
                    zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.agency_trip_ratio * accessibility)

            print("zone:", zone_obj.zone_id, " group:", grp_id,
                  " accessibility:", np.round(accessibility, 4),
                  " group_total_trips:", np.round(zone_group_total_trips, 4),
                  " group_trips_ratio:", np.round(zone_obj.zone_pop_ratio_dict[grp_id], 4),
                  " pop_weight_accessibility:",
                  np.round(zone_obj.zone_group_dict[grp_id].pop_weighted_accessibility, 4),
                  " agency_pop_weight_accessibility:",
                  np.round(zone_obj.zone_group_dict[grp_id].agency_pop_weighted_accessibility, 4))

        # calculate the accessibility of each zone
    # calculate mean accessibility
    mean_accessibility = 0
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            mean_accessibility += zone_obj.zone_group_dict[grp_id].pop_weighted_accessibility
    print("mean_accessibility: ", mean_accessibility)
    params['mean_accessibility'] = mean_accessibility

    # calculate the mean accessibility for each agency
    for agen in g_agency_list:
        agency_mean_accessibility = 0
        for zone_obj in agen.zone_list:
            for grp_id in zone_obj.zone_group_list:
                agency_mean_accessibility += zone_obj.zone_group_dict[grp_id].agency_pop_weighted_accessibility
        agen.mean_accessibility = agency_mean_accessibility
        print("agency:", agen.agency_id, " mean_accessibility:", agency_mean_accessibility)
        params[str('agency_') + str(agen.agency_id) + '_mean_accessibility'] = agency_mean_accessibility


def calculate_base_theil0_equity_index():
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            zone_obj.zone_group_dict[grp_id].base_entropy = (
                np.log(params['mean_accessibility'] / zone_obj.zone_group_dict[grp_id].accessibility))
            zone_obj.zone_group_dict[grp_id].theil_index = (zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.trip_ratio
                                                            * zone_obj.zone_group_dict[grp_id].base_entropy)
            print("zone:", zone_obj.zone_id, " group:", grp_id,
                  " base entropy:", np.round(zone_obj.zone_group_dict[grp_id].base_entropy, 4))
            print("zone:", zone_obj.zone_id, " group:", grp_id,
                  " theil index:", np.round(zone_obj.zone_group_dict[grp_id].theil_index, 4))

    # calcualte agency equity index
    for agen in g_agency_list:
        for zone_obj in agen.zone_list:
            for grp_id in zone_obj.zone_group_list:
                zone_obj.zone_group_dict[grp_id].agency_base_entropy = (
                    np.log(params[str('agency_') + str(agen.agency_id) + '_mean_accessibility'] /
                           zone_obj.zone_group_dict[grp_id].accessibility))
                zone_obj.zone_group_dict[grp_id].agency_theil_index = (zone_obj.zone_pop_ratio_dict[grp_id] *
                                                                       zone_obj.agency_trip_ratio *
                                                                       zone_obj.zone_group_dict[
                                                                           grp_id].agency_base_entropy)

                print("agency:", agen.agency_id, " zone:", zone_obj.zone_id, " group:", grp_id,
                      " base entropy:", np.round(zone_obj.zone_group_dict[grp_id].agency_base_entropy, 4))
                print("agency:", agen.agency_id, " zone:", zone_obj.zone_id, " group:", grp_id,
                      " theil index:", np.round(zone_obj.zone_group_dict[grp_id].agency_theil_index, 4))

    # calculate the equity index
    equity_index = 0
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            equity_index += zone_obj.zone_group_dict[grp_id].theil_index
    print("equity_index: ", equity_index)
    params['equity_index'] = equity_index

    # calculate the equity index for each agency
    for agen in g_agency_list:
        agency_equity_index = 0
        for zone_obj in agen.zone_list:
            for grp_id in zone_obj.zone_group_list:
                agency_equity_index += zone_obj.zone_group_dict[grp_id].agency_theil_index
        print("agency:", agen.agency_id, " equity_index:", agency_equity_index)
        params[str('agency_') + str(agen.agency_id) + '_equity_index'] = agency_equity_index


def calculate_base_theil1_equity_index():
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            zone_obj.zone_group_dict[grp_id].base_entropy = \
                (zone_obj.zone_group_dict[grp_id].accessibility
                 / params['mean_accessibility']) * \
                np.log(zone_obj.zone_group_dict[grp_id].accessibility
                       / params['mean_accessibility'])
            zone_obj.zone_group_dict[grp_id].theil_index = (zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.trip_ratio
                                                            * zone_obj.zone_group_dict[grp_id].base_entropy)
            print("zone:", zone_obj.zone_id, " group:", grp_id,
                  " base entropy:", np.round(zone_obj.zone_group_dict[grp_id].base_entropy, 4))
            print("zone:", zone_obj.zone_id, " group:", grp_id,
                  " theil index:", np.round(zone_obj.zone_group_dict[grp_id].theil_index, 4))

    # calcualte agency equity index
    for agen in g_agency_list:
        for zone_obj in agen.zone_list:
            for grp_id in zone_obj.zone_group_list:
                # bb = params[str('agency_') + str(agen.agency_id) + '_mean_accessibility']/params['mean_accessibility']
                zone_obj.zone_group_dict[grp_id].agency_base_entropy = \
                    (zone_obj.zone_group_dict[grp_id].accessibility
                     / params[str('agency_') + str(agen.agency_id) + '_mean_accessibility']) * \
                    np.log(zone_obj.zone_group_dict[grp_id].accessibility /
                           params[str('agency_') + str(agen.agency_id) + '_mean_accessibility'])
                zone_obj.zone_group_dict[grp_id].agency_theil_index = (zone_obj.zone_pop_ratio_dict[grp_id] *
                                                                       zone_obj.agency_trip_ratio *
                                                                       zone_obj.zone_group_dict[
                                                                           grp_id].agency_base_entropy)

                print("agency:", agen.agency_id, " zone:", zone_obj.zone_id, " group:", grp_id,
                      " base entropy:", np.round(zone_obj.zone_group_dict[grp_id].agency_base_entropy, 4))
                print("agency:", agen.agency_id, " zone:", zone_obj.zone_id, " group:", grp_id,
                      " theil index:", np.round(zone_obj.zone_group_dict[grp_id].agency_theil_index, 4))

    # calculate the equity index
    equity_index = 0
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            equity_index += zone_obj.zone_group_dict[grp_id].theil_index
    print("equity_index: ", equity_index)
    params['equity_index'] = equity_index

    # calculate the equity index for each agency
    for agen in g_agency_list:
        agency_equity_index = 0
        for zone_obj in agen.zone_list:
            for grp_id in zone_obj.zone_group_list:
                agency_equity_index += zone_obj.zone_group_dict[grp_id].agency_theil_index
        print("agency:", agen.agency_id, " equity_index:", agency_equity_index)
        params[str('agency_') + str(agen.agency_id) + '_equity_index'] = agency_equity_index


def calculate_base_gini_equity_index():
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            deviation = 0
            for zone_obj_1 in g_zone_list:
                for grp_id_1 in zone_obj.zone_group_list:
                    # if they are different zone_group_id
                    if zone_obj.zone_group_dict[grp_id].zone_group_id != \
                            zone_obj_1.zone_group_dict[grp_id_1].zone_group_id:
                        deviation += (zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.trip_ratio *
                                      zone_obj_1.zone_pop_ratio_dict[grp_id_1] * zone_obj_1.trip_ratio *
                                      abs(zone_obj.zone_group_dict[grp_id].accessibility -
                                          zone_obj_1.zone_group_dict[grp_id_1].accessibility))
            zone_obj.zone_group_dict[grp_id].absolute_deviation = deviation
            zone_obj.zone_group_dict[grp_id].gini_index = \
                zone_obj.zone_group_dict[grp_id].absolute_deviation / (2 * params['mean_accessibility'])

            print("zone:", zone_obj.zone_id, " group:", grp_id, " absolute deviation:", np.round(deviation, 4))
            print("zone:", zone_obj.zone_id, " group:", grp_id, " gini index:",
                  np.round(zone_obj.zone_group_dict[grp_id].gini_index, 4))

    # calculate agency equity index
    for agen in g_agency_list:
        for zone_obj in agen.zone_list:
            for grp_id in zone_obj.zone_group_list:
                deviation = 0
                for zone_obj_1 in agen.zone_list:
                    for grp_id_1 in zone_obj_1.zone_group_list:
                        # if they are different zone_group_id
                        if zone_obj.zone_group_dict[grp_id].zone_group_id != \
                                zone_obj_1.zone_group_dict[grp_id_1].zone_group_id:
                            deviation += (zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.agency_trip_ratio *
                                          zone_obj_1.zone_pop_ratio_dict[grp_id_1] * zone_obj_1.agency_trip_ratio *
                                          abs(zone_obj.zone_group_dict[grp_id].accessibility -
                                              zone_obj_1.zone_group_dict[grp_id_1].accessibility))
                zone_obj.zone_group_dict[grp_id].agency_absolute_deviation = deviation
                zone_obj.zone_group_dict[grp_id].agency_gini_index = \
                    zone_obj.zone_group_dict[grp_id].agency_absolute_deviation / \
                    (2 * params['mean_accessibility'])

                print("agency:", agen.agency_id, " zone:", zone_obj.zone_id, " group:", grp_id,
                      " absolute deviation:", np.round(deviation, 4))
                print("agency:", agen.agency_id, " zone:", zone_obj.zone_id, " group:", grp_id,
                      " gini index:", np.round(zone_obj.zone_group_dict[grp_id].agency_gini_index, 4))

    # calculate the equity index
    equity_index = 0
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            equity_index += zone_obj.zone_group_dict[grp_id].gini_index
    print("equity_index: ", equity_index)
    params['equity_index'] = equity_index

    # calculate the equity index for each agency
    for agen in g_agency_list:
        agency_equity_index = 0
        for zone_obj in agen.zone_list:
            for grp_id in zone_obj.zone_group_list:
                agency_equity_index += zone_obj.zone_group_dict[grp_id].agency_gini_index
        print("agency:", agen.agency_id, " equity_index:", agency_equity_index)
        params[str('agency_') + str(agen.agency_id) + '_equity_index'] = agency_equity_index


def calculate_base_variance_equity_index():
    # coefficient of variation
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            zone_obj.zone_group_dict[grp_id].square = \
                (zone_obj.zone_group_dict[grp_id].accessibility - params['mean_accessibility']) ** 2
            zone_obj.zone_group_dict[grp_id].variance = \
                (zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.trip_ratio
                 * zone_obj.zone_group_dict[grp_id].square)
            print("zone:", zone_obj.zone_id, " group:", grp_id,
                  " square_deviation:", np.round(zone_obj.zone_group_dict[grp_id].square, 4))
            print("zone:", zone_obj.zone_id, " group:", grp_id,
                  " variance:", np.round(zone_obj.zone_group_dict[grp_id].variance, 4))

    # calcualte agency equity index
    for agen in g_agency_list:
        for zone_obj in agen.zone_list:
            for grp_id in zone_obj.zone_group_list:
                zone_obj.zone_group_dict[grp_id].agency_square = \
                    (zone_obj.zone_group_dict[grp_id].accessibility -
                     params[str('agency_') + str(agen.agency_id) + '_mean_accessibility']) ** 2
                zone_obj.zone_group_dict[grp_id].agency_variance = (zone_obj.zone_pop_ratio_dict[grp_id] *
                                                                    zone_obj.agency_trip_ratio *
                                                                    zone_obj.zone_group_dict[grp_id].agency_square)

                print("agency:", agen.agency_id, " zone:", zone_obj.zone_id, " group:", grp_id,
                      " square_deviation:", np.round(zone_obj.zone_group_dict[grp_id].agency_square, 4))
                print("agency:", agen.agency_id, " zone:", zone_obj.zone_id, " group:", grp_id,
                      " variance:", np.round(zone_obj.zone_group_dict[grp_id].agency_variance, 4))

    # calculate the equity index
    total_deviation = 0
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            total_deviation += zone_obj.zone_group_dict[grp_id].variance
    standard_deviation = total_deviation ** 0.5
    equity_index = standard_deviation / params['mean_accessibility']
    print("equity_index: ", equity_index)
    params['equity_index'] = equity_index

    # calculate the equity index for each agency
    for agen in g_agency_list:
        agency_total_deviation = 0
        for zone_obj in agen.zone_list:
            for grp_id in zone_obj.zone_group_list:
                agency_total_deviation += zone_obj.zone_group_dict[grp_id].agency_variance
        agency_standard_deviation = agency_total_deviation ** 0.5
        agency_equity_index = \
            agency_standard_deviation / params[str('agency_') + str(agen.agency_id) + '_mean_accessibility']
        print("agency:", agen.agency_id, " equity_index:", agency_equity_index)
        params[str('agency_') + str(agen.agency_id) + '_equity_index'] = agency_equity_index


def calculate_base_atkinson_equity_index(parameter=ATK_PARAMETER):
    # coefficient of variation
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            zone_obj.zone_group_dict[grp_id].atkinson_ratio = \
                (zone_obj.zone_group_dict[grp_id].accessibility / params['mean_accessibility']) ** parameter
            zone_obj.zone_group_dict[grp_id].atkinson_ratio_p = \
                (zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.trip_ratio
                 * zone_obj.zone_group_dict[grp_id].atkinson_ratio)
            print("zone:", zone_obj.zone_id, " group:", grp_id,
                  " atkinson_ratio:", np.round(zone_obj.zone_group_dict[grp_id].atkinson_ratio, 4))
            print("zone:", zone_obj.zone_id, " group:", grp_id,
                  " atkinson_ratio_p:", np.round(zone_obj.zone_group_dict[grp_id].atkinson_ratio_p, 4))

    # calcualte agency equity index
    for agen in g_agency_list:
        for zone_obj in agen.zone_list:
            for grp_id in zone_obj.zone_group_list:
                zone_obj.zone_group_dict[grp_id].agency_atkinson_ratio = \
                    (zone_obj.zone_group_dict[grp_id].accessibility /
                     params[str('agency_') + str(agen.agency_id) + '_mean_accessibility']) ** parameter
                zone_obj.zone_group_dict[grp_id].agency_atkinson_ratio_p = \
                    (zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.agency_trip_ratio *
                     zone_obj.zone_group_dict[grp_id].agency_atkinson_ratio)

                print("agency:", agen.agency_id, " zone:", zone_obj.zone_id, " group:", grp_id,
                      " atkinson_ratio:", np.round(zone_obj.zone_group_dict[grp_id].agency_atkinson_ratio, 4))
                print("agency:", agen.agency_id, " zone:", zone_obj.zone_id, " group:", grp_id,
                      " atkinson_ratio_p:", np.round(zone_obj.zone_group_dict[grp_id].agency_atkinson_ratio_p, 4))

    # calculate the equity index
    total_ratio = 0
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            total_ratio += zone_obj.zone_group_dict[grp_id].atkinson_ratio_p
    equity_index = 1 - total_ratio ** (1 / parameter)
    print("equity_index: ", equity_index)
    params['equity_index'] = equity_index

    # calculate the equity index for each agency
    for agen in g_agency_list:
        agency_total_ratio = 0
        for zone_obj in agen.zone_list:
            for grp_id in zone_obj.zone_group_list:
                agency_total_ratio += zone_obj.zone_group_dict[grp_id].agency_atkinson_ratio_p
        agency_equity_index = 1 - agency_total_ratio ** (1 / parameter)
        print("agency:", agen.agency_id, " equity_index:", agency_equity_index)
        params[str('agency_') + str(agen.agency_id) + '_equity_index'] = agency_equity_index


def calculate_base_transit_revenue():
    # calculate the transit volume
    revenue = 0
    for od in g_demand_list:
        for od_group in od.od_group_list:
            revenue += od_group.base_transit_volume * od_group.transit_tc

    for agency in g_agency_list:
        income = 0
        for od in agency.od_list:
            for od_group in od.od_group_list:
                income += od_group.base_transit_volume * od_group.transit_tc
        agency.base_revenue = income
        params[str('agency_') + str(agency.agency_id) + '_base_transit_revenue'] = income
        print("agency:", agency.agency_id, " base_transit_revenue:", agency.base_revenue)

    params['base_transit_revenue'] = revenue
    print("base_transit_revenue: ", revenue)


def create_variables():
    for od in g_demand_list:
        for od_group in od.od_group_list:
            od_group.initialize_variables()
            od_group.subsidy()
    for agc in g_agency_list:
        g_variable_list_dict[agc.agency_id] = []
        g_gradient_list_dict[agc.agency_id] = []
        for variable in agc.var_dict.values():
            g_variable_list_dict[agc.agency_id].append(variable)
            g_gradient_list_dict[agc.agency_id].append(0)


def calculate_adjusted_prob():
    for od in g_demand_list:
        for od_group in od.od_group_list:
            od_group.calculate_adjusted_utility()
            od_group.calculate_adjusted_probability()
            od_group.calculate_adjusted_volume()


def calculate_adjusted_accessibility():
    # similar structure as the base accessibility
    for od in g_demand_list:
        for od_group in od.od_group_list:
            od_group.calculate_od_adjusted_transit_accessibility()

    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            adjusted_accessibility = 0
            for od in zone_obj.od_list:
                adjusted_accessibility += od.od_group_dict[grp_id].adjusted_od_transit_accessibility
            zone_obj.zone_group_dict[grp_id].adjusted_accessibility = adjusted_accessibility
            # assume that the trips ratio is the same as the pop ratio
            zone_obj.zone_group_dict[grp_id].adjusted_pop_weighted_accessibility = (
                    zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.trip_ratio * adjusted_accessibility)
            if log_flag:
                print("zone:", zone_obj.zone_id, " group:", grp_id,
                      " adjusted_accessibility:", adjusted_accessibility,
                      " adjusted_pop_weight_accessibility:",
                      zone_obj.zone_group_dict[grp_id].adjusted_pop_weighted_accessibility)
    adjusted_mean_accessibility = 0
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            # print(zone_obj.zone_group_dict[grp_id].adjusted_pop_weighted_accessibility)
            adjusted_mean_accessibility += zone_obj.zone_group_dict[grp_id].adjusted_pop_weighted_accessibility
    if log_flag:
        print("adjusted_mean_accessibility: ", adjusted_mean_accessibility)
    params['adjusted_mean_accessibility'] = adjusted_mean_accessibility

    # calculate the adjusted pop weighted accessibility for each agency
    for agen in g_agency_list:
        agency_adjusted_mean_accessibility = 0
        for zone_obj in agen.zone_list:
            for grp_id in zone_obj.zone_group_list:
                zone_obj.zone_group_dict[grp_id].adjusted_agency_pop_weighted_accessibility = (
                        zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.agency_trip_ratio * zone_obj.zone_group_dict[
                    grp_id].adjusted_accessibility)
                agency_adjusted_mean_accessibility += (
                    zone_obj.zone_group_dict[grp_id].adjusted_agency_pop_weighted_accessibility)

                if log_flag:
                    print("zone:", zone_obj.zone_id, " group:", grp_id,
                          " adjusted_agency_pop_weight_accessibility:",
                          zone_obj.zone_group_dict[grp_id].adjusted_agency_pop_weighted_accessibility)
        if log_flag:
            print("agency:", agen.agency_id, " adjusted_mean_accessibility:", agency_adjusted_mean_accessibility)
        params[str('agency_') + str(agen.agency_id) + '_adjusted_mean_accessibility'] = (
            agency_adjusted_mean_accessibility)


#

def calculate_adjusted_theil0_equity_index(agc_id, nb_epoch):
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            zone_obj.zone_group_dict[grp_id].adjusted_entropy = \
                (tf.math.log(params['adjusted_mean_accessibility'] /
                             zone_obj.zone_group_dict[grp_id].adjusted_accessibility))
            zone_obj.zone_group_dict[grp_id].adjusted_theil_index = (
                    zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.trip_ratio
                    * zone_obj.zone_group_dict[grp_id].adjusted_entropy)
            if log_flag:
                print("zone:", zone_obj.zone_id, " group:", grp_id,
                      " adjusted entropy:", zone_obj.zone_group_dict[grp_id].adjusted_entropy)
                print("zone:", zone_obj.zone_id, " group:", grp_id,
                      " adjusted theil index:", zone_obj.zone_group_dict[grp_id].adjusted_theil_index)
    # calculate the equity index
    adjusted_equity_index = 0
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            adjusted_equity_index += zone_obj.zone_group_dict[grp_id].adjusted_theil_index
    if log_flag:
        print("adjusted_equity_index: ", adjusted_equity_index)
    params[str(nb_epoch) + '_agency' + str(agc_id) + '_adjusted_equity_index'] = adjusted_equity_index

    # calculate the adjusted equity index for each agency
    for agen in g_agency_list:
        agency_adjusted_equity_index = 0
        for zone_obj in agen.zone_list:
            for grp_id in zone_obj.zone_group_list:
                zone_obj.zone_group_dict[grp_id].adjusted_agency_entropy = (
                    tf.math.log(params[str('agency_') + str(agen.agency_id) + '_adjusted_mean_accessibility'] /
                                zone_obj.zone_group_dict[grp_id].adjusted_accessibility))
                zone_obj.zone_group_dict[grp_id].adjusted_agency_theil_index = (
                        zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.agency_trip_ratio *
                        zone_obj.zone_group_dict[grp_id].adjusted_agency_entropy)
                if log_flag:
                    print("zone:", zone_obj.zone_id, " group:", grp_id,
                          " adjusted agency entropy:", zone_obj.zone_group_dict[grp_id].adjusted_agency_entropy)
                    print("zone:", zone_obj.zone_id, " group:", grp_id,
                          " adjusted agency theil index:", zone_obj.zone_group_dict[grp_id].adjusted_agency_theil_index)
                agency_adjusted_equity_index += zone_obj.zone_group_dict[grp_id].adjusted_agency_theil_index
        if log_flag:
            print("agency:", agen.agency_id, " adjusted equity index:", agency_adjusted_equity_index)
        params[str(nb_epoch) + '_agency' + str(agen.agency_id) + '_internal_adjusted_equity_index'] = \
            agency_adjusted_equity_index


def calculate_adjusted_theil1_equity_index(agc_id, nb_epoch):
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            zone_obj.zone_group_dict[grp_id].adjusted_entropy = \
                (zone_obj.zone_group_dict[grp_id].adjusted_accessibility
                 / params['adjusted_mean_accessibility']) * \
                (tf.math.log(zone_obj.zone_group_dict[grp_id].adjusted_accessibility
                             / params['adjusted_mean_accessibility']))

            zone_obj.zone_group_dict[grp_id].adjusted_theil_index = (
                    zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.trip_ratio
                    * zone_obj.zone_group_dict[grp_id].adjusted_entropy)
            if log_flag:
                print("zone:", zone_obj.zone_id, " group:", grp_id,
                      " adjusted entropy:", zone_obj.zone_group_dict[grp_id].adjusted_entropy)
                print("zone:", zone_obj.zone_id, " group:", grp_id,
                      " adjusted theil index:", zone_obj.zone_group_dict[grp_id].adjusted_theil_index)
    # calculate the equity index
    adjusted_equity_index = 0
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            adjusted_equity_index += zone_obj.zone_group_dict[grp_id].adjusted_theil_index
    if log_flag:
        print("adjusted_equity_index: ", adjusted_equity_index)
    params[str(nb_epoch) + '_agency' + str(agc_id) + '_adjusted_equity_index'] = adjusted_equity_index

    # calculate the adjusted equity index for each agency
    for agen in g_agency_list:
        agency_adjusted_equity_index = 0
        for zone_obj in agen.zone_list:
            for grp_id in zone_obj.zone_group_list:
                # bb = params[str('agency_') + str(agen.agency_id) + '_adjusted_mean_accessibility']/\
                #      params['adjusted_mean_accessibility']
                zone_obj.zone_group_dict[grp_id].adjusted_agency_entropy = \
                    (zone_obj.zone_group_dict[grp_id].adjusted_accessibility
                     / params[str('agency_') + str(agen.agency_id) + '_adjusted_mean_accessibility']) * \
                    (tf.math.log(zone_obj.zone_group_dict[grp_id].adjusted_accessibility
                                 / params[str('agency_') + str(agen.agency_id) + '_adjusted_mean_accessibility']))

                zone_obj.zone_group_dict[grp_id].adjusted_agency_theil_index = (
                        zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.agency_trip_ratio *
                        zone_obj.zone_group_dict[grp_id].adjusted_agency_entropy)
                if log_flag:
                    print("zone:", zone_obj.zone_id, " group:", grp_id,
                          " adjusted agency entropy:", zone_obj.zone_group_dict[grp_id].adjusted_agency_entropy)
                    print("zone:", zone_obj.zone_id, " group:", grp_id,
                          " adjusted agency theil index:", zone_obj.zone_group_dict[grp_id].adjusted_agency_theil_index)
                agency_adjusted_equity_index += zone_obj.zone_group_dict[grp_id].adjusted_agency_theil_index
        if log_flag:
            print("agency:", agen.agency_id, " adjusted equity index:", agency_adjusted_equity_index)
        params[str(nb_epoch) + '_agency' + str(agen.agency_id) + '_internal_adjusted_equity_index'] = \
            agency_adjusted_equity_index


def calculate_adjusted_gini_equity_index(agc_id, nb_epoch):
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            deviation = 0
            for zone_obj2 in g_zone_list:
                for grp_id2 in zone_obj2.zone_group_list:
                    if zone_obj.zone_group_dict[grp_id].zone_group_id != \
                            zone_obj2.zone_group_dict[grp_id2].zone_group_id:
                        deviation += (zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.trip_ratio *
                                      zone_obj2.zone_pop_ratio_dict[grp_id2] * zone_obj2.trip_ratio *
                                      abs(zone_obj.zone_group_dict[grp_id].adjusted_accessibility -
                                          zone_obj2.zone_group_dict[grp_id2].adjusted_accessibility))
            zone_obj.zone_group_dict[grp_id].adjusted_absolute_deviation = deviation
            zone_obj.zone_group_dict[grp_id].adjusted_gini_index = \
                zone_obj.zone_group_dict[grp_id].adjusted_absolute_deviation \
                / (2 * params['adjusted_mean_accessibility'])
            if log_flag:
                print("zone:", zone_obj.zone_id, " group:", grp_id,
                      " adjusted absolute deviation:", zone_obj.zone_group_dict[grp_id].adjusted_absolute_deviation)
                print("zone:", zone_obj.zone_id, " group:", grp_id,
                      " adjusted gini index:", zone_obj.zone_group_dict[grp_id].adjusted_gini_index)
    # calculate the equity index
    adjusted_equity_index = 0
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            adjusted_equity_index += zone_obj.zone_group_dict[grp_id].adjusted_gini_index
    print("adjusted_equity_index: ", adjusted_equity_index)
    params[str(nb_epoch) + '_agency' + str(agc_id) + '_adjusted_equity_index'] = adjusted_equity_index

    # calculate the adjusted equity index for each agency
    for agen in g_agency_list:
        agency_adjusted_equity_index = 0
        for zone_obj in agen.zone_list:
            for grp_id in zone_obj.zone_group_list:
                deviation = 0
                for zone_obj2 in agen.zone_list:
                    for grp_id2 in zone_obj2.zone_group_list:
                        if zone_obj.zone_group_dict[grp_id].zone_group_id != \
                                zone_obj2.zone_group_dict[grp_id2].zone_group_id:
                            deviation += (zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.agency_trip_ratio *
                                          zone_obj2.zone_pop_ratio_dict[grp_id2] * zone_obj2.agency_trip_ratio *
                                          tf.math.abs(zone_obj.zone_group_dict[grp_id].adjusted_accessibility -
                                                      zone_obj2.zone_group_dict[grp_id2].adjusted_accessibility))
                zone_obj.zone_group_dict[grp_id].adjusted_agency_absolute_deviation = deviation
                zone_obj.zone_group_dict[grp_id].adjusted_agency_gini_index = \
                    deviation / (2 * params[str('agency_') + str(agen.agency_id) + '_adjusted_mean_accessibility'])
                if log_flag:
                    print("zone:", zone_obj.zone_id, " group:", grp_id,
                          " adjusted agency absolute deviation:",
                          zone_obj.zone_group_dict[grp_id].adjusted_agency_absolute_deviation)
                    print("zone:", zone_obj.zone_id, " group:", grp_id,
                          " adjusted agency gini index:", zone_obj.zone_group_dict[grp_id].adjusted_agency_gini_index)
                agency_adjusted_equity_index += zone_obj.zone_group_dict[grp_id].adjusted_agency_gini_index
        if log_flag:
            print("agency:", agen.agency_id, " adjusted equity index:", agency_adjusted_equity_index)
        params[str(nb_epoch) + '_agency' + str(agen.agency_id) + '_internal_adjusted_equity_index'] = \
            agency_adjusted_equity_index


def calculate_adjusted_variance_equity_index(agc_id, nb_epoch):
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            zone_obj.zone_group_dict[grp_id].adjusted_square = \
                (params['adjusted_mean_accessibility'] - zone_obj.zone_group_dict[grp_id].adjusted_accessibility) ** 2
            zone_obj.zone_group_dict[grp_id].adjusted_variance = (
                    zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.trip_ratio
                    * zone_obj.zone_group_dict[grp_id].adjusted_square)
            if log_flag:
                print("zone:", zone_obj.zone_id, " group:", grp_id,
                      " square:", zone_obj.zone_group_dict[grp_id].adjusted_square)
                print("zone:", zone_obj.zone_id, " group:", grp_id,
                      " variance:", zone_obj.zone_group_dict[grp_id].adjusted_variance)

    # calculate the equity index
    adjusted_total_deviation = 0
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            adjusted_total_deviation += zone_obj.zone_group_dict[grp_id].adjusted_variance
    standard_deviation = adjusted_total_deviation ** 0.5
    adjusted_equity_index = standard_deviation / params['adjusted_mean_accessibility']
    # if log_flag:
    print("adjusted_equity_index: ", adjusted_equity_index)
    params[str(nb_epoch) + '_agency' + str(agc_id) + '_adjusted_equity_index'] = adjusted_equity_index

    # calculate the adjusted equity index for each agency
    for agen in g_agency_list:
        agency_total_deviation = 0
        for zone_obj in agen.zone_list:
            for grp_id in zone_obj.zone_group_list:
                zone_obj.zone_group_dict[grp_id].adjusted_agency_square = \
                    (zone_obj.zone_group_dict[grp_id].adjusted_accessibility -
                     params[str('agency_') + str(agen.agency_id) + '_adjusted_mean_accessibility']) ** 2
                zone_obj.zone_group_dict[grp_id].adjusted_agency_variance = (
                        zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.agency_trip_ratio *
                        zone_obj.zone_group_dict[grp_id].adjusted_agency_square)
                if log_flag:
                    print("zone:", zone_obj.zone_id, " group:", grp_id,
                          " adjusted agency square:", zone_obj.zone_group_dict[grp_id].adjusted_agency_square)
                    print("zone:", zone_obj.zone_id, " group:", grp_id,
                          " adjusted agency variance:", zone_obj.zone_group_dict[grp_id].adjusted_variance)
                agency_total_deviation += zone_obj.zone_group_dict[grp_id].adjusted_agency_variance
        agency_standard_deviation = agency_total_deviation ** 0.5
        agency_adjusted_equity_index = agency_standard_deviation / \
                                       params[str('agency_') + str(agen.agency_id) + '_adjusted_mean_accessibility']

        if log_flag:
            print("agency:", agen.agency_id, " adjusted equity index:", agency_adjusted_equity_index)
        params[str(nb_epoch) + '_agency' + str(agen.agency_id) + '_internal_adjusted_equity_index'] = \
            agency_adjusted_equity_index


def calculate_adjusted_atkinson_equity_index(agc_id, nb_epoch, parameter):
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            zone_obj.zone_group_dict[grp_id].adjusted_atkinson_ratio = \
                (zone_obj.zone_group_dict[grp_id].adjusted_accessibility /
                 params['adjusted_mean_accessibility']) ** parameter
            zone_obj.zone_group_dict[grp_id].adjusted_atkinson_ratio_p = (
                    zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.trip_ratio
                    * zone_obj.zone_group_dict[grp_id].adjusted_atkinson_ratio)
            if log_flag:
                print("zone:", zone_obj.zone_id, " group:", grp_id,
                      " adjusted_atkinson_ratio:", zone_obj.zone_group_dict[grp_id].adjusted_atkinson_ratio)
                print("zone:", zone_obj.zone_id, " group:", grp_id,
                      " adjusted_atkinson_ratio_p:", zone_obj.zone_group_dict[grp_id].adjusted_atkinson_ratio_p)

    # calculate the equity index
    adjusted_total_ratio = 0
    for zone_obj in g_zone_list:
        for grp_id in zone_obj.zone_group_list:
            adjusted_total_ratio += zone_obj.zone_group_dict[grp_id].adjusted_atkinson_ratio_p
    adjusted_equity_index = 1 - adjusted_total_ratio ** (1 / parameter)
    # if log_flag:
    print("adjusted_equity_index: ", adjusted_equity_index)
    params[str(nb_epoch) + '_agency' + str(agc_id) + '_adjusted_equity_index'] = adjusted_equity_index

    # calculate the adjusted equity index for each agency
    for agen in g_agency_list:
        adjusted_agency_total_ratio = 0
        for zone_obj in agen.zone_list:
            for grp_id in zone_obj.zone_group_list:
                zone_obj.zone_group_dict[grp_id].adjusted_atkinson_ratio = \
                    (zone_obj.zone_group_dict[grp_id].adjusted_accessibility /
                     params[str('agency_') + str(agen.agency_id) + '_adjusted_mean_accessibility']) ** parameter

                zone_obj.zone_group_dict[grp_id].adjusted_atkinson_ratio_p = (
                        zone_obj.zone_pop_ratio_dict[grp_id] * zone_obj.agency_trip_ratio *
                        zone_obj.zone_group_dict[grp_id].adjusted_atkinson_ratio)
                if log_flag:
                    print("zone:", zone_obj.zone_id, " group:", grp_id,
                          " adjusted_atkinson_ratio:", zone_obj.zone_group_dict[grp_id].adjusted_atkinson_ratio)
                    print("zone:", zone_obj.zone_id, " group:", grp_id,
                          " adjusted_atkinson_ratio_p:", zone_obj.zone_group_dict[grp_id].adjusted_atkinson_ratio_p)
                adjusted_agency_total_ratio += zone_obj.zone_group_dict[grp_id].adjusted_atkinson_ratio_p

        agency_adjusted_equity_index = 1 - adjusted_agency_total_ratio ** (1 / parameter)

        if log_flag:
            print("agency:", agen.agency_id, " adjusted equity index:", agency_adjusted_equity_index)
        params[str(nb_epoch) + '_agency' + str(agen.agency_id) + '_internal_adjusted_equity_index'] = \
            agency_adjusted_equity_index


def calculate_adjusted_revenue(agency_id, nb_epoch):
    # calculate the transit volume
    total_income = 0
    total_subsidy = 0
    agc = g_agency_dict[agency_id]
    for od in agc.od_list:
        for od_group in od.od_group_list:
            total_income += (od_group.adjusted_transit_volume * od_group.transit_tc)
            total_subsidy += (od_group.adjusted_transit_volume * od_group.transit_subsidy)
    if baseline_method == 'shared':
        equity_gap = (params['equity_index'] * params['target'] -
                      params[str(nb_epoch) + '_agency' + str(agency_id) + '_adjusted_equity_index'])
        params[str(nb_epoch) + '_agency' + str(agency_id) + '_equity_improvement'] = (
                params['equity_index'] - params[str(nb_epoch) + '_agency' + str(agency_id) + '_adjusted_equity_index'])
    if baseline_method == 'internal':
        equity_gap = (params['agency_' + str(agency_id) + '_equity_index'] * params['target'] -
                      params[str(nb_epoch) + '_agency' + str(agency_id) + '_internal_adjusted_equity_index'])
        params[str(nb_epoch) + '_agency' + str(agency_id) + '_equity_improvement'] = (
                params['agency_' + str(agency_id) + '_equity_index'] -
                params[str(nb_epoch) + '_agency' + str(agency_id) + '_internal_adjusted_equity_index'])
    params[str(nb_epoch) + '_agency' + str(agency_id) + '_equity_gap'] = equity_gap
    params[str(nb_epoch) + '_agency' + str(agency_id) + '_TEC_demand'] = equity_gap * 100000
    equity_income = equity_gap * params['TEC_price'] * 100000

    g_agency_dict[agency_id].equity_improvement = (
        params)[str(nb_epoch) + '_agency' + str(agency_id) + '_equity_improvement']
    g_agency_dict[agency_id].adjusted_equity_index = (
        params)[str(nb_epoch) + '_agency' + str(agency_id) + '_adjusted_equity_index']
    g_agency_dict[agency_id].equity_gap = equity_gap
    g_agency_dict[agency_id].tec_demand = equity_gap * 100000
    g_agency_dict[agency_id].equity_income = equity_income

    params[str(nb_epoch) + '_agency' + str(agency_id) + '_total_subsidy'] = total_subsidy
    params[str(nb_epoch) + '_agency' + str(agency_id) + '_adjusted_transit_income'] = total_income
    params[str(nb_epoch) + '_agency' + str(agency_id) + '_equity_change_income'] = equity_income
    params[str(nb_epoch) + '_agency' + str(agency_id) + '_adjusted_transit_revenue'] = (
            total_income - total_subsidy + equity_income)
    total_revenue = total_income - total_subsidy + equity_income
    g_agency_dict[agency_id].total_subsidy = total_subsidy
    g_agency_dict[agency_id].adjusted_transit_income = total_income
    g_agency_dict[agency_id].equity_change_income = equity_income
    g_agency_dict[agency_id].adjusted_transit_revenue = total_revenue

    if log_flag:
        print("agency_id: ", agency_id,
              " total_subsidy: ", total_subsidy, "\n",
              "adjusted_transit_income: ", total_income, "\n",
              "adjusted_transit_revenue: ", total_income - total_subsidy + equity_income, "\n", )
    return total_income, total_subsidy, equity_income, total_revenue


def est_gradient(variable_list, agency_id, opt, nb_epoch, method):
    # calculate the gradients
    with tf.GradientTape(persistent=True) as tape:
        calculate_adjusted_prob()
        calculate_adjusted_accessibility()
        if method == 'Generalized Entropy(1)':
            calculate_adjusted_theil1_equity_index(agency_id, nb_epoch)
        elif method == 'Gini':
            calculate_adjusted_gini_equity_index(agency_id, nb_epoch)
        elif method == 'Coefficient of Variation':
            calculate_adjusted_variance_equity_index(agency_id, nb_epoch)
        elif method == 'Atkinson':
            calculate_adjusted_atkinson_equity_index(agency_id, nb_epoch, ATK_PARAMETER)
        else:
            calculate_adjusted_theil0_equity_index(agency_id, nb_epoch)

        # calculate_adjusted_theil1_equity_index(agency_id, nb_epoch)
        t_income, t_subsidy, t_equity_income, t_revenue = calculate_adjusted_revenue(agency_id, nb_epoch)
        loss = - t_revenue
        loss_income = - t_income + t_subsidy
        loss_equity = - t_equity_income
        # calculate the gradients

    grad_income = tape.gradient(loss_income, variable_list)
    # opt.apply_gradients(zip(grad_income, variable_list))
    grad_equity = tape.gradient(loss_equity, variable_list)
    # opt.apply_gradients(zip(grad_equity, variable_list))
    grad = tape.gradient(loss, variable_list)
    opt.apply_gradients(zip(grad, variable_list))
    for variables, gg in zip(variable_list, grad):
        # var.assign_sub(gg * 0.001)
        # print(variables.name[:-2], gg.numpy())
        variables.assign(tf.clip_by_value(variables, 0, var_upper_bound[variables.name[:-2]]))
        # print(variables.name[:-2], variables.numpy())

    return grad, loss, grad_income, grad_equity
    # return loss


# ==================== Main function ====================
if __name__ == '__main__':
    # data_input
    input_path = './'
    TEC_price = 0     # per 0.00001 equity improvement
    target = 0.8
    final_target = 0.8
    # if target = final_target, it uses the fixed target method
    # if target > final_target, it uses the phased target method
    step_size = 0.001
    price_not_change_epoch = 10
    params['step_size'] = step_size
    params['TEC_price'] = TEC_price
    params['target'] = float(target)
    # equity_method = 'Gini'
    # equity_method = 'Coefficient of Variation'
    # equity_method = 'Atkinson'
    equity_method = 'Generalized Entropy(0)'
    # equity_method = 'Generalized Entropy(1)'
    # read the input data
    data_input(input_path)
    # calculate the base values
    calculate_base_probability()
    calculate_base_accessibility()
    if equity_method == 'Generalized Entropy(1)':  # Theil1
        calculate_base_theil1_equity_index()
    elif equity_method == 'Gini':  # Gini
        calculate_base_gini_equity_index()
    elif equity_method == 'Coefficient of Variation':  # Coefficient of Variation
        calculate_base_variance_equity_index()
    elif equity_method == 'Atkinson':  # Atkinson
        calculate_base_atkinson_equity_index(ATK_PARAMETER)
    else:
        calculate_base_theil0_equity_index()

    calculate_base_transit_revenue()
    create_variables()

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
    for epoch in range(total_epoch):
        print("epoch: ", epoch)
        for agency in g_agency_list:
            print("agency_id: ", agency.agency_id)
            var_list = g_variable_list_dict[agency.agency_id]
            for i in range(total_iteration):
                epoch_list.append(epoch)
                iteration_list.append(i)
                optimizer = Adam(learning_rate=0.01, epsilon=0.0001)
                gradients, loss_func, gradients_income, gradients_equity = \
                    est_gradient(var_list, agency.agency_id, optimizer, epoch, equity_method)
                loss_value_list.append(loss_func.numpy())
                agent_id_list.append(agency.agency_id)
                # calculate the mean value of the gradients
                gradients_list.append(tf.reduce_mean(gradients).numpy())
                agency.loss_func = loss_func
                gap = abs(agency.loss_func - agency.prev_loss) / abs(agency.prev_loss)
                # convert to %
                gap = gap * 100
                print("iteration: ", i, " total_revenue: ",
                      params[str(epoch) + '_agency' + str(agency.agency_id) + '_adjusted_transit_revenue'])
                print("gap: ", gap.numpy(), "%")
                total_complementary_value = 0
                for var_seq in range(len(g_variable_list_dict[agency.agency_id])):
                    var = g_variable_list_dict[agency.agency_id][var_seq]
                    var_name = var.name[:-2]
                    var_value = var.numpy().item()
                    gradients_value = gradients[var_seq].numpy().item()
                    complementary_value = abs(var_value * gradients_value)
                    grad_information_list.append([var_name, var_value, gradients_value, complementary_value,
                                                  agency.agency_id, epoch, i])
                    total_complementary_value += complementary_value
                avg_complementary = total_complementary_value / len(g_variable_list_dict[agency.agency_id])
                print("avg_complementary: ", avg_complementary)
                gap_list.append(gap.numpy())
                complementary_list.append(avg_complementary)
                if i == 0:
                    prev_complementary = avg_complementary + 1e-8
                    complementary_gap = 1
                else:
                    complementary_gap = (abs(avg_complementary - prev_complementary) / abs(prev_complementary)) * 100
                    prev_complementary = avg_complementary + 1e-8
                print("complementary_gap: ", complementary_gap, '%')
                if (gap < 1e-3) | (complementary_gap < 1e-5):
                    # if avg_complementary < prev_complementary * 0.1:
                    # if gap < 0.00001:
                    break
                agency.prev_loss = loss_func
                # print("gradients: ", gradients)
            for agc in g_agency_list:
                for var in g_variable_list_dict[agency.agency_id]:
                    print(var.name[:-2], np.round(var.numpy().item(), 4))
                    var_information_list.append([var.name[:-2], var.numpy().item(), agency.agency_id, epoch])

            print(" total_subsidy: ",
                  params[str(epoch) + '_agency' + str(agency.agency_id) + '_total_subsidy'].numpy(), "\n")

            print("base_equity_index: ", params['equity_index'])
            print("adjusted_equity_index: ",
                  params[str(epoch) + '_agency' + str(agency.agency_id) + '_adjusted_equity_index'].numpy())
            current_adjusted_equity_index = \
                params[str(epoch) + '_agency' + str(agency.agency_id) + '_adjusted_equity_index'].numpy()
            print("equity_improvement: ",
                  params[str(epoch) + '_agency' + str(agency.agency_id) + '_equity_improvement'].numpy())
            print("equity_gap: ", params[str(epoch) + '_agency' + str(agency.agency_id) + '_equity_gap'].numpy())
            print("TEC_demand: ", params[str(epoch) + '_agency' + str(agency.agency_id) + '_TEC_demand'].numpy())
            current_tec_demand = params[str(epoch) + '_agency' + str(agency.agency_id) + '_TEC_demand'].numpy()
            # if params[str(epoch) + '_agency' + str(agency.agency_id) + '_equity_gap'].numpy() > 0:
            #     params['target'] = max(params['target'] - 0.05, final_target)
            params['target'] = final_target + (target - final_target) * np.exp(-epoch / 10)
            agency_info = [epoch, agency.agency_id, params['target'],
                           params[str(epoch) + '_agency' + str(agency.agency_id) + '_adjusted_transit_revenue'].numpy(),
                           params[str(epoch) + '_agency' + str(agency.agency_id) + '_total_subsidy'].numpy(),
                           params['equity_index'],
                           params['agency_' + str(agency.agency_id) + '_equity_index'],
                           params[str(epoch) + '_agency' + str(agency.agency_id) +
                                  '_internal_adjusted_equity_index'].numpy(),
                           params[str(epoch) + '_agency' + str(agency.agency_id) + '_adjusted_equity_index'].numpy(),
                           params[str(epoch) + '_agency' + str(agency.agency_id) + '_equity_improvement'].numpy(),
                           params[str(epoch) + '_agency' + str(agency.agency_id) + '_equity_gap'].numpy(),
                           current_tec_demand,
                           params['TEC_price']]
            agency_information_list.append(agency_info)
            params['TEC_price'] = (params['TEC_price'] -
                                   params['step_size'] *
                                   current_tec_demand)
            params['TEC_price'] = max(0.0, params['TEC_price'])
            print("TEC_price: ", params['TEC_price'])
            print("====================================================================")
            # if price does not change for given iterations, then stop
        # if abs(params['TEC_price'] - TEC_price) < 0.0001:
        #     current_price_not_change_epoch += 1
        #     if current_price_not_change_epoch > price_not_change_epoch:
        #         break
        # stop the loop epoch
        if abs(np.round(current_tec_demand, 0) * params['TEC_price']) < 0.0001:
            current_price_not_change_epoch += 1
            if current_price_not_change_epoch > price_not_change_epoch:
                break
        TEC_price = params['TEC_price']

    # export the results to csv
    with open(input_path + 'output_var.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['variable', 'value', 'agency_id', 'epoch'])
        for var_info in var_information_list:
            writer.writerow(var_info)

    with open(input_path + 'output_grad.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['variable', 'value', 'gradients', 'complementary', 'agency_id', 'epoch', 'iteration'])
        for var_info in grad_information_list:
            writer.writerow(var_info)

    # write the accessibility to csv
    with open(input_path + 'output_accessibility.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['zone', 'group', 'base_accessibility', 'adjusted_accessibility'])
        for zone in g_zone_list:
            for group_id in zone.zone_group_list:
                writer.writerow([zone.zone_id, group_id, zone.zone_group_dict[group_id].accessibility,
                                 zone.zone_group_dict[group_id].adjusted_accessibility.numpy()])

    # export dictionary params to csv
    with open(input_path + 'output_params.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['param', 'value'])
        for key, value in params.items():
            writer.writerow([key, value])

    # write iteration results to csv
    with open(input_path + 'output_iteration.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'iteration', 'loss', 'gap', 'gradients', 'complementary', 'agency_id'])
        for i in range(len(loss_value_list)):
            writer.writerow([epoch_list[i], iteration_list[i], loss_value_list[i], gap_list[i], gradients_list[i],
                             complementary_list[i], agent_id_list[i]])

    # write the TEC price to csv
    with open(input_path + 'output_agency_info.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'agency_id', 'Target', 'adjusted_transit_revenue', 'total_subsidy',
                         'base_equity_index', 'agency_base_equity_index', 'agency_internal_adjusted_equity_index',
                         'adjusted_equity_index', 'equity_improvement', 'equity_gap', 'TEC_demand', 'TEC_price'])
        for i in range(len(agency_information_list)):
            writer.writerow(agency_information_list[i])

    df = pd.read_csv(input_path + 'output_agency_info.csv')
    grouped = df.groupby('agency_id')
    for name, group in grouped:
        group.to_csv(input_path + 'output_agency_info_' + str(name) + '.csv', index=False)