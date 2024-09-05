import pandas as pd

pd.set_option('display.max_columns', None)

# show all columns

income_df = pd.read_csv('zone_information/zone_income_pop_ratio.csv')
pop_ratio1_dict = dict(zip(income_df['GEOID'], income_df['minc<25k']))
pop_ratio2_dict = dict(zip(income_df['GEOID'], income_df['minc25k_50k']))
pop_ratio3_dict = dict(zip(income_df['GEOID'], income_df['minc50k_100k']))
pop_ratio4_dict = dict(zip(income_df['GEOID'], income_df['minc>100k']))
geoid_income_dict = dict(zip(income_df['GEOID'], income_df['hhminc']))

auto_node_df = pd.read_csv('auto_network/node.csv')
auto_node_df = auto_node_df[auto_node_df['node_type'] == 'centroid']
auto_centroid_df = auto_node_df.copy()
# convert auto centroid name to int
auto_centroid_df.loc[:, 'name'] = auto_centroid_df.loc[:, 'name'].apply(lambda x: int(x))
auto_centroid_df.loc[:, 'node_id'] = auto_centroid_df.loc[:, 'node_id'].apply(lambda x: int(x))
auto_centroid_dict = dict(zip(auto_centroid_df['name'], auto_centroid_df['node_id']))
auto_result_df = pd.read_csv('auto_network/result.csv')
auto_result_df['od_pair'] = auto_result_df.apply(lambda x: (int(x.ozone_id), int(x.dzone_id)), axis=1)
auto_travel_time_dict = dict(zip(auto_result_df['od_pair'], auto_result_df['travel_time']))
auto_cost_dict = dict(zip(auto_result_df['od_pair'], auto_result_df['cost']))

transit_node_df = pd.read_csv('transit_network/node.csv')
transit_node_df = transit_node_df[transit_node_df['node_type'] == 'centroid']
transit_centroid_df = transit_node_df.copy()
# convert transit centroid name to int
transit_centroid_df.loc[:, 'name'] = transit_centroid_df['name'].apply(lambda x: int(x))
transit_centroid_df.loc[:, 'node_id'] = transit_centroid_df['node_id'].apply(lambda x: int(x))
transit_centroid_dict = dict(zip(transit_centroid_df['name'], transit_centroid_df['node_id']))
transit_result_df = pd.read_csv('transit_network/result.csv')
transit_result_df['od_pair'] = transit_result_df.apply(lambda x: (int(x.ozone_id), int(x.dzone_id)), axis=1)
transit_travel_time_dict = dict(zip(transit_result_df['od_pair'], transit_result_df['travel_time']))
transit_cost_dict = dict(zip(transit_result_df['od_pair'], transit_result_df['cost']))
transit_check_flag_dict = dict(zip(transit_result_df['od_pair'], transit_result_df['check_flag']))
# read auto_network
zone_df = pd.read_csv('zone_data/zone.csv')
# zone_id	x_coord	y_coord	geometry	pop	home_work	pop_ratio_1	pop_ratio_2	pop_ratio_3	pop_ratio_4	pop_ratio_5
# pop_ratio_6	pop_ratio_7	pop_ratio_8	pop_ratio_9	pop_ratio_10	agency_id	trip_rate	total_trip
zone_df = zone_df[['GEOID', 'INTPTLAT', 'INTPTLON']]
zone_df = zone_df.rename(columns={'GEOID': 'zone_id'})
zone_df = zone_df.rename(columns={'INTPTLAT': 'y_coord'})
zone_df = zone_df.rename(columns={'INTPTLON': 'x_coord'})
zone_df['geometry'] = \
    zone_df.apply(lambda row: "POINT (" + str(row['x_coord']) + " " + str(row['y_coord']) + ")", axis=1)

zone_performance_df = pd.read_csv('zone_information/zone_performance.csv')
geoid_pop_dict = dict(zip(zone_performance_df['trctfp'], zone_performance_df['totPop']))
# geoid_income_dict = dict(zip(zone_performance_df['trctfp'], zone_performance_df['hhminc']))

zone_df['pop'] = zone_df['zone_id'].apply(lambda x: geoid_pop_dict.setdefault(x, -1))
zone_df['est_flag'] = zone_df['pop'].apply(lambda x: 1 if x == -1 else 0)
mean_pop = zone_df[zone_df['pop'] != -1]['pop'].mean()
zone_df['pop'] = zone_df['pop'].apply(lambda x: x if (x != -1) & (x != 0) else mean_pop)

zone_df['hhminc'] = zone_df['zone_id'].apply(lambda x: geoid_income_dict.setdefault(x, -1))
# calculate the mean value of hhminc excluding nan pop weighted
temp_df = zone_df[zone_df['hhminc'] != -1]
temp_df = temp_df[temp_df['pop'] != -1]
mean_hhminc = (temp_df['hhminc'] * temp_df['pop']).sum() / temp_df['pop'].sum()
zone_df['hhminc'] = zone_df['hhminc'].apply(lambda x: x if (x != -1) & (x != 0) else mean_hhminc)

total_mean = income_df['minc<25k'].mean() + income_df['minc25k_50k'].mean() + \
             income_df['minc50k_100k'].mean() + income_df['minc>100k'].mean()

pop_ratio1_mean = income_df['minc<25k'].mean() / total_mean
pop_ratio2_mean = income_df['minc25k_50k'].mean() / total_mean
pop_ratio3_mean = income_df['minc50k_100k'].mean() / total_mean
pop_ratio4_mean = income_df['minc>100k'].mean() / total_mean

zone_df['pop_ratio_1'] = zone_df['zone_id'].apply(lambda x: pop_ratio1_dict.setdefault(x, pop_ratio1_mean))
zone_df['pop_ratio_2'] = zone_df['zone_id'].apply(lambda x: pop_ratio2_dict.setdefault(x, pop_ratio2_mean))
zone_df['pop_ratio_3'] = zone_df['zone_id'].apply(lambda x: pop_ratio3_dict.setdefault(x, pop_ratio3_mean))
zone_df['pop_ratio_4'] = zone_df['zone_id'].apply(lambda x: pop_ratio4_dict.setdefault(x, pop_ratio4_mean))
zone_df['pop_ratio_1'] = zone_df['pop_ratio_1']/100
zone_df['pop_ratio_2'] = zone_df['pop_ratio_2']/100
zone_df['pop_ratio_3'] = zone_df['pop_ratio_3']/100
zone_df['pop_ratio_4'] = zone_df['pop_ratio_4']/100
zone_df['agency_id'] = 1
zone_df['trip_rate'] = 1.25
zone_df['total_trip'] = zone_df['pop'] * zone_df['trip_rate']

# demand_id	from_zone_id	to_zone_id	travel_time_auto	travel_time_transit	travel_cost_auto
# travel_cost_transit	od_split_rate	temp_total_sum	trips
zone_df = zone_df[['zone_id', 'pop', 'hhminc', 'pop_ratio_1', 'pop_ratio_2', 'pop_ratio_3', 'pop_ratio_4',
                   'agency_id', 'trip_rate', 'total_trip', 'est_flag', 'x_coord', 'y_coord',
                   'geometry']]
zone_df.to_csv('zone.csv', index=False)

demand_seq = 0
demand_list = []
for _, row in zone_df.iterrows():
    for _, p_row in zone_df.iterrows():
        from_transit_node_id = transit_centroid_dict[row['zone_id']]
        to_transit_node_id = transit_centroid_dict[p_row['zone_id']]
        check_flag = transit_check_flag_dict.setdefault((from_transit_node_id, to_transit_node_id), 0)
        if (row['zone_id'] != p_row['zone_id']) & (check_flag == 1):
            demand_id = demand_seq
            from_zone_id = row['zone_id']
            to_zone_id = p_row['zone_id']
            from_auto_node_id = auto_centroid_dict[row['zone_id']]
            to_auto_node_id = auto_centroid_dict[p_row['zone_id']]
            # from_transit_node_id = transit_centroid_dict[row['zone_id']]
            # to_transit_node_id = transit_centroid_dict[p_row['zone_id']]
            travel_time_auto = auto_travel_time_dict.setdefault((from_auto_node_id, to_auto_node_id), 120)
            travel_time_transit = transit_travel_time_dict.setdefault((from_transit_node_id, to_transit_node_id), 120)
            travel_cost_auto = auto_cost_dict.setdefault((from_auto_node_id, to_auto_node_id), 5)
            travel_cost_transit = transit_cost_dict.setdefault((from_transit_node_id, to_transit_node_id), 5)
            # gravity model
            from_zone_trip = row['total_trip']
            to_zone_trip = p_row['total_trip']
            geometry = "LINESTRING (" + str(row['x_coord']) + " " + str(row['y_coord']) + ", " + \
                       str(p_row['x_coord']) + " " + str(p_row['y_coord']) + ")"

            generalized_cost = (60 / 25) * (
                    travel_cost_auto + travel_cost_transit) + travel_time_transit + travel_time_auto
            demand_list.append([demand_id, from_zone_id, to_zone_id, travel_time_auto, travel_time_transit,
                                travel_cost_auto, travel_cost_transit, from_zone_trip, to_zone_trip, generalized_cost,
                                geometry])
            demand_seq += 1
            if demand_seq % 1000 == 0:
                print("number of demand pairs: ", demand_seq)

demand_df = pd.DataFrame(demand_list, columns=['demand_id', 'from_zone_id', 'to_zone_id', 'travel_time_auto',
                                               'travel_time_transit', 'travel_cost_auto', 'travel_cost_transit',
                                               'from_zone_trip', 'to_zone_trip', 'generalized_cost',
                                               'geometry'])

# assume that gravity model parameter are known, alpha = 1 and beta = 2
zone_demand_df = demand_df.groupby('from_zone_id')
total_demand_df = pd.DataFrame()
for zone_id, group_df in zone_demand_df:
    group_df['od_split_rate'] = group_df['to_zone_trip'] / (group_df['generalized_cost'] ** 2)
    group_df['od_split_rate'] = group_df['od_split_rate'] / group_df['od_split_rate'].sum()
    group_df['trips'] = group_df['od_split_rate'] * group_df['from_zone_trip']
    total_demand_df = pd.concat([total_demand_df, group_df])

# convert minutes to hours
total_demand_df['travel_time_auto'] = total_demand_df['travel_time_auto'] / 60
total_demand_df['travel_time_transit'] = total_demand_df['travel_time_transit'] / 60

total_demand_df = total_demand_df[['demand_id', 'from_zone_id', 'to_zone_id', 'travel_time_auto',
                                   'travel_time_transit', 'travel_cost_auto', 'travel_cost_transit',
                                   'od_split_rate', 'trips', 'geometry']]

average_vehicle_price = 20700
depreciation_rate = 0.4
average_trip_number_per_day = 2.5
average_keeping_years = 4

depreciation_cost = average_vehicle_price * depreciation_rate
depreciation_cost_per_day = depreciation_cost / average_keeping_years / 365
depreciation_cost_per_trip = depreciation_cost_per_day / average_trip_number_per_day

average_insurance_cost_per_month = 80
average_insurance_cost_per_day = average_insurance_cost_per_month / 30
average_insurance_cost_per_trip = average_insurance_cost_per_day / average_trip_number_per_day

total_demand_df['travel_cost_auto'] = total_demand_df['travel_cost_auto'] + depreciation_cost_per_trip + \
                                        average_insurance_cost_per_trip

# if 0, fill with 2.0
total_demand_df['travel_cost_transit'] = total_demand_df['travel_cost_transit'].apply(lambda x: 2.0 if x == 0 else x)

# in Sioux Falls, the maximum transit cost is 3 dollars
total_demand_df['travel_cost_transit'] = total_demand_df['travel_cost_transit'].apply(lambda x: 3.0 if x > 3.0 else x)

total_demand_df.to_csv('demand.csv', index=False)
