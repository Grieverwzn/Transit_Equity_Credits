# -*- coding: utf-8 -*-
import copy
import os
import math
import datetime
import numpy as np
import pandas as pd
import time

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

"""-------------------main steps------------------"""


def reading_data(gtfs_path):
    print('start reading input GTFS files...')
    agency_df = _reading_text(gtfs_path + os.sep + 'agency')
    agency_name = agency_df['agency_name'][0]
    if '"' in agency_name:
        agency_name = eval(agency_name)
        #  Remove quotes from string, for example '"Arlington Transit"' --> 'Arlington Transit'
    print("agent_name:", agency_name)

    stop_df = _reading_text(gtfs_path + os.sep + 'stops')
    stop_df = stop_df[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]
    print("number of stops =", len(stop_df))
    #  select only latitude and longitude of the stops/stations

    route_df = _reading_text(gtfs_path + os.sep + 'routes')
    route_df = route_df[['route_id', 'route_short_name', 'route_long_name', 'route_type']]
    print("number of routes =", len(route_df))

    trip_df = _reading_text(gtfs_path + os.sep + 'trips')
    if 'direction_id' not in trip_df.columns.tolist():  # direction_id is mandatory field name here
        trip_df['direction_id'] = str(0)
    trip_df['direction_id'] = trip_df.apply(lambda x: str(2 - int(x['direction_id'])), axis=1)
    directed_route_id = trip_df['route_id'].astype(str).str.cat(
        trip_df['direction_id'].astype(str), sep='.')
    trip_df['directed_route_id'] = directed_route_id  # add a new column "directed_route_id"
    #  If trips on a route service opposite directions,distinguish directions using values 0 and 1.
    # revise the direction_id from 0,1 to 2,1
    # add a new field directed_route_id
    # deal with special issues of Agency 12 Fairfax CUE # Alicia, Nov 10:
    # route file has route id with quotes, e.g., '"green2"' while trip file does not have it, e.g.,'green2'
    if (route_df['route_id'][0][0] == '"') != (trip_df['route_id'][0][0] == '"'):
        if route_df['route_id'][0][0] == '"':
            route_df['route_id'] = route_df.apply(lambda x: x['route_id'].strip('"'), axis=1)
        else:
            trip_df['route_id'] = trip_df.apply(lambda x: x['route_id'].strip('"'), axis=1)
    trip_route_df = pd.merge(trip_df, route_df, on='route_id')  # might go wrong in Agency 12
    print("number of trips =", len(trip_route_df), "...", len(trip_df))
    #  as route is higher level planning than trips, len(trip_route_df)=len(trip_df)

    stop_time_df = _reading_text(gtfs_path + os.sep + 'stop_times')
    print("number of stop_time records =", len(stop_time_df))
    # drop the stations without accurate arrival and departure time.

    # drop nan
    stop_time_df = stop_time_df.dropna(subset=['arrival_time'], how='any')
    # drop ''
    stop_time_df = stop_time_df[stop_time_df.arrival_time != '']
    stop_time_df = stop_time_df[stop_time_df.departure_time != '']

    # drop ' '
    stop_time_df = stop_time_df[stop_time_df.arrival_time != ' ']
    stop_time_df = stop_time_df[stop_time_df.departure_time != ' ']
    print("number of stop_time records after dropping empty arrival and departure time =", len(stop_time_df))

    # convert timestamp to minute
    # as some agencies might have trips overlapping two days, should use _to_timedelta to convert the data
    print("start converting the time stamps...")
    tt = datetime.datetime(2021, 1, 1, 0, 0, 0, 0)
    stop_time_df['arrival_time'] = pd.to_timedelta(stop_time_df['arrival_time']) + tt
    stop_time_df['departure_time'] = pd.to_timedelta(stop_time_df['departure_time']) + tt
    stop_time_df['arrival_time'] = \
        stop_time_df['arrival_time'].apply(lambda x: x.hour * 60 + x.minute + 1440 * (x.day - 1))
    stop_time_df['departure_time'] = \
        stop_time_df['departure_time'].apply(lambda x: x.hour * 60 + x.minute + 1440 * (x.day - 1))

    print("start marking terminal flags for stops...")
    iteration_group = stop_time_df.groupby(['trip_id'])
    # mark terminal flag for each stop. The terminals can only be determined at the level of trips
    input_list = []
    time_start = time.time()
    for trip_id, trip_stop_time_df in iteration_group: # FutureWarning: In a future version of pandas, a length 1 tuple will be returned when iterating over a groupby with a grouper equal to a list of length 1. Don't supply a list with a single grouper to avoid this warning.
        trip_stop_time_df = trip_stop_time_df.sort_values(by=['stop_sequence'])
        trip_stop_time_df = trip_stop_time_df.reset_index()
        # select only the trips within the provided time window
        if (trip_stop_time_df.arrival_time.min() <= period_end_time) & (
                trip_stop_time_df.arrival_time.min() >= period_start_time):
            input_list.append(trip_stop_time_df)
    intermediate_output_list = list(map(_determine_terminal_flag, input_list))
    output_list = list(map(_stop_sequence_label, intermediate_output_list))

    # use map function to speed up marking process
    time_end = time.time()
    print('add terminal_flag for trips using CPU time:', time_end - time_start, 's')

    time_start = time.time()
    stop_time_df_with_terminal = pd.concat(output_list, axis=0)
    # concatenating a list is much faster than concatenating separate dataframes
    time_end = time.time()
    print('concatenate different trips using CPU time:', time_end - time_start, 's')
    print("have updated", len(stop_time_df_with_terminal), "stop_time records")
    print("merge the route information with trip information...")
    directed_trip_route_stop_time_df = pd.merge(trip_route_df, stop_time_df_with_terminal, on='trip_id')
    print("number of final merged records =", len(directed_trip_route_stop_time_df))
    print("Data reading done..")

    #  as trip is higher level planning than stop time scheduling, len(stop_time_df)>=len(trip_df)
    #  Each record of directed_trip_route_stop_time_df represents a space-time state of a vehicle
    # trip_id (different vehicles, e.g., train lines)
    # stop_id (spatial location of the vehicle)
    # arrival_time,departure_time (time index of the vehicle)

    directed_route_stop_id = directed_trip_route_stop_time_df['directed_route_id'].astype(
        str).str.cat(directed_trip_route_stop_time_df['stop_id'].astype(str), sep='.')
    directed_trip_route_stop_time_df['directed_route_stop_id'] = directed_route_stop_id
    #  directed_route_stop_id is a unique id to identify the route, direction, and stop of a vehicle at a time point
    directed_trip_route_stop_time_df['stop_sequence'] \
        = directed_trip_route_stop_time_df['stop_sequence'].astype('int32')
    # two important concepts : 1 directed_service_stop_id (directed_route_stop_id + stop sequence)
    directed_trip_route_stop_time_df['directed_service_stop_id'] = \
        directed_trip_route_stop_time_df.directed_route_stop_id.astype(str) + ':' + \
        directed_trip_route_stop_time_df.stop_sequence_label
    # 2. directed service id (directed_route_id + stop sequence) same directed route id might have different sequences
    directed_trip_route_stop_time_df['directed_service_id'] = \
        directed_trip_route_stop_time_df.directed_route_id.astype(str) + ':' + \
        directed_trip_route_stop_time_df.stop_sequence_label
    #  attach stop name and geometry for stops
    directed_trip_route_stop_time_df = pd.merge(directed_trip_route_stop_time_df, stop_df, on='stop_id')
    directed_trip_route_stop_time_df['agency_name'] = agency_name

    return stop_df, route_df, trip_df, trip_route_df, stop_time_df, directed_trip_route_stop_time_df


def create_nodes(directed_trip_route_stop_time_df, agency_num):
    """create physical (station) node..."""
    physical_node_df = pd.DataFrame()
    temp_df = directed_trip_route_stop_time_df.drop_duplicates(subset=['stop_id'])
    physical_node_df['name'] = temp_df['stop_id']
    physical_node_df = physical_node_df.sort_values(by=['name'])
    physical_node_df['node_id'] = \
        np.linspace(start=1, stop=len(physical_node_df), num=len(physical_node_df)).astype('int32')
    physical_node_df['node_id'] += int('{}000000'.format(agency_num))
    physical_node_df['physical_node_id'] = physical_node_df['node_id']
    physical_node_df['x_coord'] = temp_df['stop_lon'].astype(float)
    physical_node_df['y_coord'] = temp_df['stop_lat'].astype(float)
    physical_node_df['route_type'] = temp_df['route_type']
    physical_node_df['route_id'] = temp_df['route_id']
    physical_node_df['node_type'] = \
        physical_node_df.apply(lambda x: _convert_route_type_to_node_type_p(x.route_type), axis=1)
    physical_node_df['directed_route_id'] = ""
    physical_node_df['directed_service_id'] = ""
    physical_node_df['zone_id'] = ""
    physical_node_df['agency_name'] = temp_df['agency_name']
    physical_node_df['geometry'] = 'POINT (' + physical_node_df['x_coord'].astype(str) + \
                                   ' ' + physical_node_df['y_coord'].astype(str) + ')'
    stop_name_id_dict = dict(zip(physical_node_df['name'], physical_node_df['node_id']))
    physical_node_df['terminal_flag'] = temp_df['terminal_flag']
    physical_node_df['ctrl_type'] = ""
    physical_node_df['agent_type'] = ""

    """ create service node..."""
    service_node_df = pd.DataFrame()
    temp_df = directed_trip_route_stop_time_df.drop_duplicates(subset=['directed_service_stop_id'])
    # 2.2.2 route stop node
    service_node_df['name'] = temp_df['directed_service_stop_id']
    service_node_df = service_node_df.sort_values(by=['name'])
    service_node_df['node_id'] = \
        np.linspace(start=1, stop=len(service_node_df), num=len(service_node_df)).astype('int32')
    service_node_df['physical_node_id'] = temp_df.apply(lambda x: stop_name_id_dict[x.stop_id], axis=1)
    service_node_df['node_id'] += int('{}500000'.format(agency_num))

    service_node_df['x_coord'] = temp_df['stop_lon'].astype(float) - 0.000100
    service_node_df['y_coord'] = temp_df['stop_lat'].astype(float) - 0.000100
    service_node_df['route_type'] = temp_df['route_type']
    service_node_df['route_id'] = temp_df['route_id']
    service_node_df['node_type'] = \
        service_node_df.apply(lambda x: _convert_route_type_to_node_type_s(x.route_type), axis=1)
    # node_csv['terminal_flag'] = ' '
    service_node_df['directed_route_id'] = temp_df['directed_route_id'].astype(str)
    service_node_df['directed_service_id'] = temp_df['directed_service_id'].astype(str)
    service_node_df['zone_id'] = ""
    service_node_df['agency_name'] = temp_df['agency_name']
    service_node_df['geometry'] = \
        'POINT (' + service_node_df['x_coord'].astype(str) + ' ' + service_node_df['y_coord'].astype(str) + ')'

    service_node_df['terminal_flag'] = temp_df['terminal_flag']
    service_node_df['ctrl_type'] = ""
    service_node_df['agent_type'] = ""
    # concatenate service and physical node
    node_df = pd.concat([physical_node_df, service_node_df])
    return node_df


def create_service_boarding_links(directed_trip_route_stop_time_df, node_df, agency_num, one_agency_link_list):
    """dictionaries"""
    node_id_dict = dict(zip(node_df['name'], node_df['node_id']))
    directed_service_dict = dict(zip(node_df['node_id'], node_df['name']))
    node_lon_dict = dict(zip(node_df['node_id'], node_df['x_coord']))
    node_lat_dict = dict(zip(node_df['node_id'], node_df['y_coord']))
    frequency_dict = {}

    print("1. start creating route links...")
    """service links"""
    number_of_route_links = 0
    iteration_group = directed_trip_route_stop_time_df.groupby('directed_service_id')
    labeled_directed_service_list = []

    time_start = time.time()
    for directed_service_id, route_df in iteration_group:
        if directed_service_id in labeled_directed_service_list:
            continue
        else:
            labeled_directed_service_list.append(directed_service_id)
            number_of_trips = len(route_df.trip_id.unique())
            frequency_dict[directed_service_id] = number_of_trips  # note the frequency of routes
            one_line_df = route_df[route_df.trip_id == route_df.trip_id.unique()[0]]
            one_line_df = one_line_df.sort_values(by=['stop_sequence'])
            number_of_records = len(one_line_df)
            one_line_df = one_line_df.reset_index()
            for k in range(number_of_records - 1):
                link_id = 1000000 * agency_num + number_of_route_links + 1
                from_node_id = node_id_dict[one_line_df.iloc[k].directed_service_stop_id]
                to_node_id = node_id_dict[one_line_df.iloc[k + 1].directed_service_stop_id]
                facility_type = _convert_route_type_to_link_type(one_line_df.iloc[k].route_type)
                # print(one_line_df.iloc[k].route_type)
                dir_flag = 1
                directed_route_id = one_line_df.iloc[k].directed_route_id
                link_type = 1
                link_type_name = 'service_links'
                # from_node_lon = float(one_line_df.iloc[k].stop_lon) - 0.000100
                from_node_lon = node_lon_dict[from_node_id]
                # from_node_lat = float(one_line_df.iloc[k].stop_lat) - 0.000100
                from_node_lat = node_lat_dict[from_node_id]
                # to_node_lon = float(one_line_df.iloc[k + 1].stop_lon) - 0.000100
                # to_node_lat = float(one_line_df.iloc[k + 1].stop_lat)- 0.000100
                to_node_lon = node_lon_dict[to_node_id]
                to_node_lat = node_lat_dict[to_node_id]
                length = _calculate_distance_from_geometry(from_node_lon, from_node_lat, to_node_lon, to_node_lat)
                lanes = number_of_trips
                capacity = 999999
                VDF_fftt1 = one_line_df.iloc[k + 1].arrival_time - one_line_df.iloc[k].arrival_time
                if VDF_fftt1 == 0:
                    VDF_fftt1 = 0.5  # since the unit of VDF_fftt1 is minutes, we set the minimum value to 0.5 minutes
                # minutes
                VDF_cap1 = lanes * capacity
                free_speed = ((length / 1000) / VDF_fftt1) * 60
                # (kilometers/minutes)*60 = kilometer/hour
                VDF_alpha1 = 0.15
                VDF_beta1 = 4
                VDF_penalty1 = 0
                cost = 0
                geometry = 'LINESTRING (' + str(from_node_lon) + ' ' + str(from_node_lat) + ', ' + \
                           str(to_node_lon) + ' ' + str(to_node_lat) + ')'
                agency_name = one_line_df.agency_name[0]
                allowed_use = _allowed_use_function(one_line_df.iloc[k].route_type)
                stop_sequence = one_line_df.iloc[k].stop_sequence
                directed_service_id = one_line_df.iloc[k].directed_service_id
                link_list = [link_id, from_node_id, to_node_id, facility_type, dir_flag, directed_route_id,
                             link_type, link_type_name, length, lanes, capacity, free_speed, cost,
                             VDF_fftt1, VDF_cap1, VDF_alpha1, VDF_beta1, VDF_penalty1, geometry, allowed_use,
                             agency_name,
                             stop_sequence, directed_service_id]
                one_agency_link_list.append(link_list)
                number_of_route_links += 1
                if number_of_route_links % 50 == 0:
                    time_end = time.time()
                    print('convert ', number_of_route_links,
                          'service links successfully...', 'using time', time_end - time_start, 's')

    print("2. start creating boarding links from stations to their passing routes...")
    """boarding_links"""
    service_node_df = node_df[node_df.node_id != node_df.physical_node_id]
    #  select service node from node_df
    service_node_df = service_node_df.reset_index()
    number_of_sta2route_links = 0
    for iter, row in service_node_df.iterrows():
        link_id = agency_num * 1000000 + number_of_route_links + number_of_sta2route_links
        from_node_id = row.physical_node_id
        to_node_id = row.node_id
        facility_type = _convert_route_type_to_link_type(row.route_type)
        dir_flag = 1
        directed_route_id = row.directed_route_id
        link_type = 2
        link_type_name = 'boarding_links'
        to_node_lon = row.x_coord
        to_node_lat = row.y_coord
        from_node_lon = node_lon_dict[row.physical_node_id]
        from_node_lat = node_lat_dict[row.physical_node_id]
        # length = _calculate_distance_from_geometry(from_node_lon, from_node_lat, to_node_lon, to_node_lat)
        length = 50
        free_speed = ((length / 1000) / 2) * 60  # (kilometers/minutes)*60 = kilometer/hour
        lanes = 1
        capacity = 999999
        VDF_cap1 = lanes * capacity
        VDF_alpha1 = 0.15
        VDF_beta1 = 4
        VDF_penalty1 = 0
        cost = _calculate_ticket_cost(facility_type)
        stop_sequence = -1
        directed_service_id = directed_service_dict[to_node_id]
        geometry = 'LINESTRING (' + str(from_node_lon) + ' ' + str(from_node_lat) + ', ' + \
                   str(to_node_lon) + ' ' + str(to_node_lat) + ')'
        agency_name = row.agency_name
        allowed_use = _allowed_use_function(row.route_type)

        # inbound links (boarding)

        VDF_fftt1 = \
            0.5 * ((period_end_time - period_start_time) / frequency_dict[row.directed_service_id]) + 2
        VDF_fftt1 = min(VDF_fftt1, 2) # waiting time at a station is 2 minutes at most
        # waiting time at a station is 30 minutes at most
        geometry = 'LINESTRING (' + str(to_node_lon) + ' ' + str(to_node_lat) + ', ' + \
                   str(from_node_lon) + ' ' + str(from_node_lat) + ')'
        # inbound link is average waiting time derived from frequency
        link_list_inbound = [link_id, from_node_id, to_node_id, facility_type, dir_flag, directed_route_id,
                             link_type, link_type_name, length, lanes, capacity, free_speed, cost,
                             VDF_fftt1, VDF_cap1, VDF_alpha1, VDF_beta1, VDF_penalty1, geometry, allowed_use,
                             agency_name,
                             stop_sequence, directed_service_id]
        number_of_sta2route_links += 1

        # outbound links (boarding)
        link_id = agency_num * 1000000 + number_of_route_links + number_of_sta2route_links
        VDF_fftt1 = 2  # (length / free_speed) * 60
        #  the time of outbound time
        cost = 0
        link_list_outbound = [link_id, to_node_id, from_node_id, facility_type, dir_flag, directed_route_id,
                              link_type, link_type_name, length, lanes, capacity, free_speed, cost,
                              VDF_fftt1, VDF_cap1, VDF_alpha1, VDF_beta1, VDF_penalty1, geometry, allowed_use,
                              agency_name,
                              stop_sequence, directed_service_id]
        one_agency_link_list.append(link_list_inbound)
        one_agency_link_list.append(link_list_outbound)
        number_of_sta2route_links += 1
        #  one inbound link and one outbound link
        if number_of_sta2route_links % 50 == 0:
            time_end = time.time()
            print('convert ', number_of_sta2route_links,
                  'boarding links successfully...', 'using time', time_end - time_start, 's')

    return one_agency_link_list


def create_transferring_links(all_node_df, all_link_list):
    physical_node_df = all_node_df[all_node_df.node_id == all_node_df.physical_node_id]
    physical_node_df = physical_node_df.reset_index()
    number_of_transferring_links = 0
    time_start = time.time()
    for i in range(len(physical_node_df)):
        ref_x = physical_node_df.iloc[i].x_coord
        ref_y = physical_node_df.iloc[i].y_coord
        neighboring_node_df = physical_node_df[(physical_node_df.x_coord >= (ref_x - 0.003)) &
                                               (physical_node_df.x_coord <= (ref_x + 0.003))]
        neighboring_node_df = neighboring_node_df[(neighboring_node_df.y_coord >= (ref_y - 0.003)) &
                                                  (neighboring_node_df.y_coord <= (ref_y + 0.003))]
        labeled_list = []
        count = 0
        for j in range(len(neighboring_node_df)):
            if count >= 10:
                break
            if (physical_node_df.iloc[i].route_id, physical_node_df.iloc[i].agency_name) == \
                    (neighboring_node_df.iloc[j].route_id, neighboring_node_df.iloc[j].agency_name):
                continue
            from_node_lon = float(physical_node_df.iloc[i].x_coord)
            from_node_lat = float(physical_node_df.iloc[i].y_coord)
            to_node_lon = float(neighboring_node_df.iloc[j].x_coord)
            to_node_lat = float(neighboring_node_df.iloc[j].y_coord)
            length = _calculate_distance_from_geometry(from_node_lon, from_node_lat, to_node_lon, to_node_lat)
            if (length > 321.869) | (length < 1):
                continue
            if (neighboring_node_df.iloc[j].route_id, neighboring_node_df.iloc[j].agency_name) in labeled_list:
                continue
            count += 1
            labeled_list.append((neighboring_node_df.iloc[j].route_id, neighboring_node_df.iloc[j].agency_name))
            # consider only one stops of another route
            # transferring 1
            #  print('transferring link length =', length)
            link_id = number_of_transferring_links + 1
            from_node_id = physical_node_df.iloc[i].node_id
            to_node_id = neighboring_node_df.iloc[j].node_id
            facility_type = 'sta2sta'
            dir_flag = 1
            directed_route_id = -1
            link_type = 3
            link_type_name = 'transferring_links'
            lanes = 1
            capacity = 999999
            VDF_fftt1 = (length / 1000) / 1
            VDF_cap1 = lanes * capacity
            free_speed = 1
            # 1 kilo/hour
            VDF_alpha1 = 0.15
            VDF_beta1 = 4
            VDF_penalty1 = _transferring_penalty(physical_node_df.iloc[i].node_type,
                                                 neighboring_node_df.iloc[j].node_type)
            # penalty of transferring
            cost = 0
            geometry = 'LINESTRING (' + str(from_node_lon) + ' ' + str(from_node_lat) + ', ' + \
                       str(to_node_lon) + ' ' + str(to_node_lat) + ')'
            agency_name = ""
            allowed_use = \
                _allowed_use_transferring(physical_node_df.iloc[i].node_type, neighboring_node_df.iloc[j].node_type)
            stop_sequence = ""
            directed_service_id = ""
            link_list = [link_id, from_node_id, to_node_id, facility_type, dir_flag, directed_route_id,
                         link_type, link_type_name, length, lanes, capacity, free_speed, cost,
                         VDF_fftt1, VDF_cap1, VDF_alpha1, VDF_beta1, VDF_penalty1, geometry, allowed_use, agency_name,
                         stop_sequence, directed_service_id]
            all_link_list.append(link_list)
            # transferring 2
            number_of_transferring_links += 1
            geometry = 'LINESTRING (' + str(to_node_lon) + ' ' + str(to_node_lat) + ', ' + \
                       str(from_node_lon) + ' ' + str(from_node_lat) + ')'
            link_id = number_of_transferring_links + 1
            link_list = [link_id, to_node_id, from_node_id, facility_type, dir_flag, directed_route_id,
                         link_type, link_type_name, length, lanes, capacity, free_speed, cost,
                         VDF_fftt1, VDF_cap1, VDF_alpha1, VDF_beta1, VDF_penalty1, geometry, allowed_use, agency_name,
                         stop_sequence, directed_service_id]
            all_link_list.append(link_list)
            number_of_transferring_links += 1
            if number_of_transferring_links % 50 == 0:
                time_end = time.time()
                print('convert ', number_of_transferring_links,
                      'transferring links successfully...', 'using time', time_end - time_start, 's')

    return all_link_list


""" ------------------functions------------------ """


def _stop_sequence_label(trip_stop_time_df):
    trip_stop_time_df = trip_stop_time_df.sort_values(by=['stop_sequence'])
    trip_stop_time_df['stop_sequence_label'] = ';'.join(np.array(trip_stop_time_df.stop_sequence).astype(str))
    return trip_stop_time_df


def _reading_text(filename):
    file_path = filename + '.txt'
    data = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        first_line = lines[0].split('\n')[0].split(',')
        for line in lines:
            if len(line.split('\n')[0].split(',')) == len(first_line):
                data.append(line.split('\n')[0].split(','))
            else:
                data.append(_split_ignore_separators_in_quoted(line))
    data_frame = pd.DataFrame(data[1:], columns=data[0])
    return data_frame


def _determine_terminal_flag(trip_stop_time_df):
    trip_stop_time_df.stop_sequence = trip_stop_time_df.stop_sequence.astype('int32')
    start_stop_seq = int(trip_stop_time_df.stop_sequence.min())
    end_stop_seq = int(trip_stop_time_df.stop_sequence.max())
    #  convert string to integer
    trip_stop_time_df['terminal_flag'] = \
        ((trip_stop_time_df.stop_sequence == start_stop_seq) |
         (trip_stop_time_df.stop_sequence == end_stop_seq)).astype('int32')
    return trip_stop_time_df


def _allowed_use_function(route_type):
    #  convert route type to node type on service network
    allowed_use = ""
    if int(route_type) == 0:
        # tram
        allowed_use = "w_bus_only;w_bus_metro;d_bus_only;d_bus_metro"
    if int(route_type) == 1:
        # metro
        allowed_use = "w_metro_only;w_bus_metro;d_metro_only;d_bus_metro"
    if int(route_type) == 2:
        # rail
        allowed_use = "w_rail_only;d_rail_only"
    if int(route_type) == 3:
        # bus
        allowed_use = "w_bus_only;w_bus_metro;d_bus_only;d_bus_metro"
    return allowed_use


def _allowed_use_transferring(node_type_1, node_type_2):
    if (node_type_1 == 'stop') & (node_type_2 == 'stop'):
        allowed_use = "w_bus_only;d_bus_only"
    elif (node_type_1 == 'stop') & (node_type_2 == 'metro_station'):
        allowed_use = "w_bus_metro;d_bus_metro"
    elif (node_type_1 == 'metro_station') & (node_type_2 == 'stop'):
        allowed_use = "w_bus_metro;d_bus_metro"
    elif (node_type_1 == 'metro_station') & (node_type_2 == 'metro_station'):
        allowed_use = "w_metro_only;d_metro_only"
    elif (node_type_1 == 'rail_station') & (node_type_2 == 'rail_station'):
        allowed_use = "w_rail_only;d_rail_only"
    else:
        allowed_use = "closed"

    return allowed_use


def _transferring_penalty(node_type_1, node_type_2):
    if (node_type_1 == 'bus_stop') & (node_type_2 == 'bus_stop'):
        VDF_penalty1 = 15
    elif (node_type_1 == 'bus_stop') & (node_type_2 == 'tram_stop'):
        VDF_penalty1 = 15
    elif (node_type_1 == 'tram_stop') & (node_type_2 == 'bus_stop'):
        VDF_penalty1 = 15
    elif (node_type_1 == 'bus_stop') & (node_type_2 == 'metro_station'):
        VDF_penalty1 = 15
    elif (node_type_1 == 'metro_station') & (node_type_2 == 'bus_stop'):
        VDF_penalty1 = 15
    elif (node_type_1 == 'metro_station') & (node_type_2 == 'metro_station'):
        VDF_penalty1 = 15
    elif (node_type_1 == 'rail_station') & (node_type_2 == 'rail_station'):
        VDF_penalty1 = 15
    else:
        VDF_penalty1 = 15

    return VDF_penalty1


def _convert_route_type_to_node_type_p(route_type):
    #  convert route type to node type on physical network
    node_type = ""
    if int(route_type) == 0:
        # tram
        node_type = 'tram_stop'
    if int(route_type) == 1:
        # metro
        node_type = 'metro_station'
    if int(route_type) == 2:
        # rail
        node_type = 'rail_station'
    if int(route_type) == 3:
        # bus
        node_type = 'bus_stop'
    if int(route_type) == 4:
        # bus
        node_type = 'ferry_port'
    if int(route_type) == 5:
        # ferry
        node_type = 'cable_tram_station'
    if int(route_type) == 11:
        # Trolleybus
        node_type = 'trolley_bus_stop'
    return node_type


def _calculate_ticket_cost(facility_type):
    cost = 0
    if facility_type == 'tram':
        cost = 1.5
    if facility_type == 'metro':
        cost = 1.5
    if facility_type == 'rail':
        cost = 1.5
    if facility_type == 'bus':
        cost = 1.5
    if facility_type == 'ferry':
        cost = 1.5
    if facility_type == 'cable_tram':
        cost = 1.5
    if facility_type == 'trolley_bus':
        cost = 1.5
    return cost


# 0 - Tram, Streetcar, Light rail. Any light rail or street level system within a metropolitan area.
# 1 - Subway, Metro. Any underground rail system within a metropolitan area.
# 2 - Rail. Used for intercity or long-distance travel.
# 3 - Bus. Used for short- and long-distance bus routes.
# 4 - Ferry. Used for short- and long-distance boat service.
# 5 - Cable tram. Used for street-level rail cars where the cable runs beneath the vehicle (e.g., cable car in San Francisco).
# 6 - Aerial lift, suspended cable car (e.g., gondola lift, aerial tramway). Cable transport where cabins, cars, gondolas or open chairs are suspended by means of one or more cables.
# 7 - Funicular. Any rail system designed for steep inclines.
# 11 - Trolleybus. Electric buses that draw power from overhead wires using poles.
# 12 - Monorail. Railway in which the track consists of a single rail or a beam.


def _convert_route_type_to_node_type_s(route_type):
    #  convert route type to node type on service network
    node_type = ""
    if int(route_type) == 0:
        # tram
        node_type = 'tram_service_node'
    if int(route_type) == 1:
        # metro
        node_type = 'metro_service_node'
    if int(route_type) == 2:
        # rail
        node_type = 'rail_service_node'
    if int(route_type) == 3:
        # bus
        node_type = 'bus_service_node'
    if int(route_type) == 4:
        # bus
        node_type = 'ferry_service_node'
    if int(route_type) == 5:
        # ferry
        node_type = 'cable_tram_service_node'
    if int(route_type) == 11:
        # Trolleybus
        node_type = 'trolley_bus_service_node'
    return node_type


def _convert_route_type_to_link_type(route_type):
    #  convert route type to node type on service network
    link_type = ""
    if int(route_type) == 0:
        # tram
        link_type = 'tram'
    if int(route_type) == 1:
        # metro
        link_type = 'metro'
    if int(route_type) == 2:
        # rail
        link_type = 'rail'
    if int(route_type) == 3:
        # bus
        link_type = 'bus'
    if int(route_type) == 4:
        # bus
        link_type = 'ferry'
    if int(route_type) == 5:
        # ferry
        link_type = 'cable_tram'
    if int(route_type) == 11:
        # Trolleybus
        link_type = 'trolley_bus'
    return link_type


def _split_ignore_separators_in_quoted(s, separator=',', quote_mark='"'):
    result = []
    quoted = False
    current = ''
    for i in range(len(s)):
        if quoted:
            current += s[i]
            if s[i] == quote_mark:
                quoted = False
            continue
        if s[i] == separator:
            result.append(current.strip())
            current = ''
        else:
            current += s[i]
            if s[i] == quote_mark:
                quoted = True
    result.append(current)
    return result


def _calculate_distance_from_geometry(lon1, lat1, lon2, lat2):  # WGS84 transfer coordinate system to distance(mile) #xy
    radius = 6371
    d_latitude = (lat2 - lat1) * math.pi / 180.0
    d_longitude = (lon2 - lon1) * math.pi / 180.0

    a = math.sin(d_latitude / 2) * math.sin(d_latitude / 2) + math.cos(lat1 * math.pi / 180.0) * math.cos(
        lat2 * math.pi / 180.0) * math.sin(d_longitude / 2) * math.sin(d_longitude / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    # distance = radius * c * 1000 / 1609.34  # mile
    distance = radius * c * 1000  # meter
    return distance


def _hhmm_to_minutes(time_period_1):
    from_time_1 = datetime.time(int(time_period_1[0:2]), int(time_period_1[2:4]))
    to_time_1 = datetime.time(int(time_period_1[-4:-2]), int(time_period_1[-2:]))
    from_time_min_1 = from_time_1.hour * 60 + from_time_1.minute
    to_time_min_1 = to_time_1.hour * 60 + to_time_1.minute
    return from_time_min_1, to_time_min_1


""" ------------------main functions------------------ """


def gtfs2gmns(input_path, output_path):
    start_time = time.time()
    folders = os.listdir(input_path)
    gtfs_folder_list = []
    for sub_folder in folders:
        sub_folder_path = input_path + '/' + sub_folder
        if os.path.isdir(sub_folder_path):  # check whether the specified path is an existing directory or not.
            gtfs_folder_list.append(sub_folder_path)
    if len(gtfs_folder_list) == 0:
        gtfs_folder_list.append(input_path)

    all_node_list = []
    all_link_list = []
    for i in range(len(gtfs_folder_list)):
        print('Start converting Agency_{}...'.format(i + 1))
        print('Directory : ' + str(gtfs_folder_list[i]))
        agency_gtfs_path = gtfs_folder_list[i]
        """ step 1. reading data """
        stop_df, route_df, trip_df, trip_route_df, stop_time_df, directed_trip_route_stop_time_df = \
            reading_data(agency_gtfs_path)
        #  directed_trip_route_stop_time_df.to_csv(gtfs_folder_list[i] + '/timetable.csv', index=False)
        #  directed_trip_route_stop_time_df = pd.read_csv(gtfs_folder_list[i] + '/timetable.csv')

        """step 2. create nodes"""
        agency_num = i + 1
        # number of agency equals to i+1
        node_df = create_nodes(directed_trip_route_stop_time_df, agency_num)
        all_node_list.append(node_df)
        print("node.csv of", str(gtfs_folder_list[i]), "has been generated...")
        """step 3. create links"""
        all_link_list \
            = create_service_boarding_links(directed_trip_route_stop_time_df, node_df, agency_num, all_link_list)

        if i == len(gtfs_folder_list):
            print('output')
        print('Conversion of  Agency{}...'.format(agency_num + 1), ' have done..')

    all_node_df = pd.concat(all_node_list)
    all_node_df.reset_index(inplace=True)
    all_node_df = all_node_df.drop(['index'], axis=1)
    # transferring links
    all_link_list = create_transferring_links(all_node_df, all_link_list)

    all_link_df = pd.DataFrame(all_link_list)
    all_link_df.rename(columns={0: 'link_id',
                                1: 'from_node_id',
                                2: 'to_node_id',
                                3: 'facility_type',
                                4: 'dir_flag',
                                5: 'directed_route_id',
                                6: 'link_type',
                                7: 'link_type_name',
                                8: 'length',
                                9: 'lanes',
                                10: 'capacity',
                                11: 'free_speed',
                                12: 'cost',
                                13: 'VDF_fftt1',
                                14: 'VDF_cap1',
                                15: 'VDF_alpha1',
                                16: 'VDF_beta1',
                                17: 'VDF_penalty1',
                                18: 'geometry',
                                19: 'VDF_allowed_uses1',
                                20: 'agency_name',
                                21: 'stop_sequence',
                                22: 'directed_service_id'}, inplace=True)
    all_node_df.to_csv('source_node.csv', index=False)
    # zone_df = pd.read_csv('zone.csv')
    #  source_node_df = pd.read_csv('source_node.csv')
    # node_df = pd.concat([zone_df, all_node_df])
    node_df = all_node_df
    node_df.to_csv("node.csv", index=False)
    all_link_df = all_link_df.drop_duplicates(
        subset=['from_node_id', 'to_node_id'],
        keep='last').reset_index(drop=True)
    all_link_df.to_csv('link.csv', index=False)
    print('run time -->', time.time() - start_time)


if __name__ == '__main__':
    global period_start_time
    global period_end_time
    input_gtfs_path = 'GTFS_SIOUXFALLS'
    output_gmns_path = '.'
    time_period_id = 1
    time_period = '0800_1000'
    period_start_time, period_end_time = _hhmm_to_minutes(time_period)

    gtfs2gmns(input_gtfs_path, output_gmns_path)
