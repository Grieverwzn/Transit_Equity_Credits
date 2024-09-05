import pandas as pd
from shapely import wkt
import geopandas as gpd
import math
import time as time


# # read new_link.csv
# new_link_df = pd.read_csv('new_link.csv')
# # from_node_id = 1
# new_link_df = new_link_df[new_link_df['from_node_id'] == 1]
# new_link_df.to_csv("check.csv", index=False)

# exit()
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


# show all columns
pd.set_option('display.max_columns', None)

zone_performance_df = pd.read_csv('zone_information/zone_performance.csv')
name_dict = dict(zip(zone_performance_df['trctfp'], zone_performance_df['pctnvh']))

# Read raw zone data
zone_df = pd.read_csv('zone_data/zone.csv')

# step 1: generate zone.csv
# change the column name order GEOID, INTPTLAT, INTPTLON
zone_df = zone_df[['GEOID', 'INTPTLAT', 'INTPTLON', 'STATEFP', 'COUNTYFP', 'TRACTCE', 'NAME']]
zone_df['pctnvh'] = zone_df['GEOID'].map(name_dict)

# fill pctnvh with mean value if it is null
mean_pctnvh = zone_df[zone_df['pctnvh'].notnull()]['pctnvh'].mean()
zone_df['pctnvh'] = zone_df['pctnvh'].fillna(mean_pctnvh)


# change name
# GEOID -> name
# INTPTLON -> x_coord
# INTPTLAT -> y_coord
zone_df = zone_df.rename(columns={'GEOID': 'name'})
zone_df['node_id'] = zone_df.index + 1
zone_df['physical_node_id'] = -1
zone_df = zone_df.rename(columns={'INTPTLAT': 'y_coord'})
zone_df = zone_df.rename(columns={'INTPTLON': 'x_coord'})
zone_df['route_type'] = 100  # 100 means centroid
zone_df['route_id'] = -1
zone_df['node_type'] = 'centroid'
zone_df['directed_route_id'] = ''
zone_df['directed_service_id'] = ''
zone_df['zone_id'] = zone_df['node_id']
zone_df['agency_name'] = ''
zone_df['geometry'] = zone_df.apply(lambda x: wkt.loads(f'POINT({x.x_coord} {x.y_coord})'), axis=1)
zone_df['terminal_flag'] = 0  # 2 means centroid
zone_df['ctrl_type'] = ''
# pctnvh -> agent_type
zone_df = zone_df.rename(columns={'pctnvh': 'agent_type'})

# make the columns in the same order
zone_df = zone_df[['name', 'node_id', 'physical_node_id', 'x_coord', 'y_coord',
                   'route_type', 'route_id', 'node_type', 'directed_route_id',
                   'directed_service_id', 'zone_id', 'agency_name', 'geometry',
                   'terminal_flag', 'ctrl_type', 'agent_type']]
print("Number of zones: ", len(zone_df))
zone_df.to_csv('transit_network/zone.csv', index=False)

agent_type_dict = dict(zip(zone_df['node_id'], zone_df['agent_type']))

# step 2: generate new_node.csv
node_df = pd.read_csv('transit_network/network_before_attaching/node.csv')
# concatenate the node_df and zone_df
node_df = pd.concat([node_df, zone_df])

print("Number of nodes: ", len(node_df))

# step 3: create a new link between centroid and other nodes where terminal_flag = 1
centroid_df = node_df[node_df['node_type'] == 'centroid']
physical_node_df = node_df[node_df['node_id'] == node_df['physical_node_id']]
print("Number of centroids: ", len(centroid_df))
print("Number of physical nodes: ", len(physical_node_df))

# for each centroid, find the nearest physical node
link_list = []
link_seq = 0  # link id
# link_id	from_node_id	to_node_id	facility_type	dir_flag
# directed_route_id	link_type	link_type_name	length	lanes
# capacity	free_speed	cost	VDF_fftt1	VDF_cap1
# VDF_alpha1	VDF_beta1	VDF_penalty1	geometry
# VDF_allowed_uses1	agency_name	stop_sequence
# directed_service_id

start = time.time()
for index, row in centroid_df.iterrows():
    nvh_ratio = min(agent_type_dict[row['node_id']], 100)  # 100%
    min_distance = (100 - nvh_ratio) * 10 + 500
    print("create centroid connector links for centroid", index, "with node_id", row['node_id'])
    # generate a bounding box of 5 km around the centroid
    # bounding box: [minx, miny, maxx, maxy]
    minx = row['x_coord'] - 0.045
    maxx = row['x_coord'] + 0.045
    miny = row['y_coord'] - 0.045
    maxy = row['y_coord'] + 0.045
    reduced_physical_node_df = physical_node_df[(physical_node_df['x_coord'] >= minx) &
                                                (physical_node_df['x_coord'] <= maxx) &
                                                (physical_node_df['y_coord'] >= miny) &
                                                (physical_node_df['y_coord'] <= maxy)]
    neighbor_distance_flag = False
    for p_index, p_row in reduced_physical_node_df.iterrows():
        distance = _calculate_distance_from_geometry(row['x_coord'], row['y_coord'], p_row['x_coord'], p_row['y_coord'])
        # print('distance from', row['node_id'], 'to', p_row['node_id'], 'is', distance)
        if distance < min_distance:
            neighbor_distance_flag = True
            # link_id, from_node_id, to_node_id, distance
            link_id = link_seq
            from_node_id = row['node_id']
            to_node_id = p_row['node_id']
            facility_type = 'centroid_connector_outward'
            dir_flag = 1
            directed_route_id = ''
            link_type = 0
            link_type_name = 'centroid_connector'
            length = distance  # meter
            lanes = 1
            capacity = 999999
            free_speed = 5  # 5 km/h driving to station
            cost = 0
            VDF_fftt1 = (length / 1000) / free_speed * 60  # minute
            VDF_cap1 = lanes * capacity
            VDF_alpha1 = 0.15
            VDF_beta1 = 4
            VDF_penalty1 = 15
            geometry = "LINESTRING(" + str(row['x_coord']) + " " + str(row['y_coord']) + "," + str(
                p_row['x_coord']) + " " + str(p_row['y_coord']) + ")"
            VDF_allowed_uses1 = ''
            agency_name = ''
            stop_sequence = ''
            directed_service_id = ''
            link_list.append([link_id, from_node_id, to_node_id, facility_type, dir_flag, directed_route_id, link_type,
                              link_type_name, length, lanes, capacity, free_speed, cost, VDF_fftt1, VDF_cap1,
                              VDF_alpha1, VDF_beta1, VDF_penalty1, geometry,
                              VDF_allowed_uses1, agency_name, stop_sequence, directed_service_id])
            link_seq += 1
            link_id = link_seq
            facility_type = 'centroid_connector_inward'
            geometry = "LINESTRING(" + str(p_row['x_coord']) + " " + str(p_row['y_coord']) + "," + str(
                row['x_coord']) + " " + str(row['y_coord']) + ")"
            link_list.append([link_id, to_node_id, from_node_id, facility_type, dir_flag, directed_route_id, link_type,
                              link_type_name, length, lanes, capacity, free_speed, cost, VDF_fftt1, VDF_cap1,
                              VDF_alpha1, VDF_beta1, VDF_penalty1, geometry,
                              VDF_allowed_uses1, agency_name, stop_sequence, directed_service_id])
            link_seq += 1
            # if link_seq % 10000 == 0:
            #     print("create ", link_seq, 'centroid connector links')
    if not neighbor_distance_flag:
        min_distance = min_distance * 1.5
        for p_index, p_row in reduced_physical_node_df.iterrows():
            distance = _calculate_distance_from_geometry(row['x_coord'], row['y_coord'], p_row['x_coord'],
                                                         p_row['y_coord'])
            # print('distance from', row['node_id'], 'to', p_row['node_id'], 'is', distance)
            neighbor_distance_flag = False
            if distance < min_distance:
                neighbor_distance_flag = True
                # link_id, from_node_id, to_node_id, distance
                link_id = link_seq
                from_node_id = row['node_id']
                to_node_id = p_row['node_id']
                facility_type = 'centroid_connector_outward'
                dir_flag = 1
                directed_route_id = ''
                link_type = 0
                link_type_name = 'centroid_connector'
                length = distance  # meter
                lanes = 1
                capacity = 999999
                free_speed = 5  # 5 km/h driving to station
                cost = 0
                VDF_fftt1 = (length / 1000) / free_speed * 60  # minute
                VDF_cap1 = lanes * capacity
                VDF_alpha1 = 0.15
                VDF_beta1 = 4
                VDF_penalty1 = 15
                geometry = "LINESTRING(" + str(row['x_coord']) + " " + str(row['y_coord']) + "," + str(
                    p_row['x_coord']) + " " + str(p_row['y_coord']) + ")"
                VDF_allowed_uses1 = ''
                agency_name = ''
                stop_sequence = ''
                directed_service_id = ''
                link_list.append(
                    [link_id, from_node_id, to_node_id, facility_type, dir_flag, directed_route_id, link_type,
                     link_type_name, length, lanes, capacity, free_speed, cost, VDF_fftt1, VDF_cap1,
                     VDF_alpha1, VDF_beta1, VDF_penalty1, geometry,
                     VDF_allowed_uses1, agency_name, stop_sequence, directed_service_id])
                link_seq += 1
                link_id = link_seq
                facility_type = 'centroid_connector_inward'
                geometry = "LINESTRING(" + str(p_row['x_coord']) + " " + str(p_row['y_coord']) + "," + str(
                    row['x_coord']) + " " + str(row['y_coord']) + ")"
                link_list.append(
                    [link_id, to_node_id, from_node_id, facility_type, dir_flag, directed_route_id, link_type,
                     link_type_name, length, lanes, capacity, free_speed, cost, VDF_fftt1, VDF_cap1,
                     VDF_alpha1, VDF_beta1, VDF_penalty1, geometry,
                     VDF_allowed_uses1, agency_name, stop_sequence, directed_service_id])
                link_seq += 1

# create link_df
link_df = pd.DataFrame(link_list, columns=['link_id', 'from_node_id', 'to_node_id', 'facility_type', 'dir_flag',
                                           'directed_route_id', 'link_type', 'link_type_name', 'length', 'lanes',
                                           'capacity', 'free_speed', 'cost', 'VDF_fftt1', 'VDF_cap1', 'VDF_alpha1',
                                           'VDF_beta1', 'VDF_penalty1', 'geometry', 'VDF_allowed_uses1', 'agency_name',
                                           'stop_sequence', 'directed_service_id'])

end = time.time()
print("Time: ", end - start)
# read link.csv
old_link_df = pd.read_csv('transit_network/network_before_attaching/link.csv')
print("Number of links: ", len(link_df))
# concatenate the link_df and link.csv
link_df = pd.concat([old_link_df, link_df])
# convert free_speed from km/h to mile/h
# link_df['free_speed'] = link_df['free_speed'] / 1.60934
# # convert length from meter to mile
# link_df['length'] = link_df['length'] / 1609.34
# # convert fftt from second to hour
# link_df['VDF_fftt1'] = link_df['VDF_fftt1'] / 3600
link_df.to_csv('transit_network/link.csv', index=False)

# step 4: # calculate the degree of each node using link_df
# calculate the degree of each node and add the degree to the node_df
out_degree = link_df.groupby('from_node_id').size().reset_index(name='out_degree')
in_degree = link_df.groupby('to_node_id').size().reset_index(name='in_degree')
# change from_node_id to node_id
out_degree = out_degree.rename(columns={'from_node_id': 'node_id'})
# change to_node_id to node_id
in_degree = in_degree.rename(columns={'to_node_id': 'node_id'})

node_df = pd.merge(node_df, out_degree, how='left', on='node_id')
node_df = pd.merge(node_df, in_degree, how='left', on='node_id')
node_df['degree'] = node_df['out_degree'] + node_df['in_degree']
node_df.to_csv('transit_network/node.csv', index=False)
print("Number of nodes: ", len(node_df))

# step 5 generate demand.csv
centroid_df = node_df[node_df['node_type'] == 'centroid']
# out_degree > 0
centroid_df = centroid_df[centroid_df['out_degree'] > 0]
print("Number of centroids with out_degree > 0: ", len(centroid_df))
demand_list = []
demand_seq = 0
# demand_id, from_node_id, to_node_id, volume
for index, row in centroid_df.iterrows():
    for p_index, p_row in centroid_df.iterrows():
        if row['node_id'] != p_row['node_id']:
            demand_id = demand_seq
            from_node_id = row['node_id']
            to_node_id = p_row['node_id']
            volume = 1
            demand_list.append([demand_id, from_node_id, to_node_id, volume])
            demand_seq += 1
            if demand_seq % 1000 == 0:
                print("create ", demand_seq, 'demands')

demand_df = pd.DataFrame(demand_list, columns=['demand_id', 'ozone_id', 'dzone_id', 'volume'])
demand_df.to_csv('transit_network/demand.csv', index=False)
