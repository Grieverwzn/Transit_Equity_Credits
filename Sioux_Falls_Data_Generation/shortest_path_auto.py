import networkx as nx
import pandas as pd
import time
# parallel computing
import multiprocessing as mp
import os

number_od_pair = 0
G = nx.DiGraph()
if os.path.exists("auto_network/graph.gpickle"):
    G = nx.read_gpickle("auto_network/graph.gpickle")


# Create a directed graph
def create_graph(all_link_df):
    # Add edges from the DataFrame
    nb_links = 0
    for _, row in all_link_df.iterrows():
        # if link_type_name is transferring_links
        travel_time = row['VDF_fftt1']
        fuel_cost = row['cost']
        weight = travel_time + fuel_cost * (60 / 25)  # (60 / 25) is the (1 hour / value of time) in minutes
        G.add_edge(row['from_node_id'], row['to_node_id'], weight=weight, cost=fuel_cost, fftt=travel_time)
        nb_links += 1
        if nb_links % 50000 == 0:
            print("Added", nb_links, "links")
    # save graph
    nx.write_gpickle(G, "auto_network/graph.gpickle")


def calculate_shortest_path(pair, output_dict):
    start_node = pair[0]
    end_node = pair[1]
    start = time.time()
    try:
        path_sum = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
        generalized_cost_sum = sum(G[path_sum[ind]][path_sum[ind + 1]]['weight'] for ind in range(len(path_sum) - 1))
        fuel_cost_sum = sum(G[path_sum[ind]][path_sum[ind + 1]]['cost'] for ind in range(len(path_sum) - 1))
        travel_time_sum = sum(G[path_sum[ind]][path_sum[ind + 1]]['fftt'] for ind in range(len(path_sum) - 1))

        output_dict[(start_node, end_node)] = \
            (path_sum, generalized_cost_sum, fuel_cost_sum, travel_time_sum)

    except nx.NetworkXNoPath:
        output_dict[(start_node, end_node)] = (None, None, None, None)
        # print("No path from", start_node, "to", end_node)
    end = time.time()
    print("CPU Time: ", end - start)


if __name__ == '__main__':
    # read graph
    node_df = pd.read_csv('auto_network/node.csv')
    link_df = pd.read_csv('auto_network/link.csv')
    # set G as global variable
    # create graph
    if not os.path.exists("auto_network/graph.gpickle"):
        create_graph(link_df)

    # read graph
    # Specify your start and end nodes using demand.csv
    demand_df = pd.read_csv('auto_network/demand.csv')
    demand_df = demand_df.sample(frac=1.0)
    print("Number of demand pairs: ", len(demand_df))

    od_pairs = [(row['ozone_id'], row['dzone_id']) for _, row in demand_df.iterrows()]
    # multiprocessing
    manager = mp.Manager()
    return_dict = manager.dict()
    # number of processes
    nb_processes = int(os.cpu_count())
    print("Number of processes: ", nb_processes)
    pool = mp.Pool(nb_processes)
    # Create a pool of processes
    start_time = time.time()
    with mp.Pool(processes=nb_processes) as pool:
        # Map node pairs to processes
        pool.starmap(calculate_shortest_path, [(pair, return_dict) for pair in od_pairs])
    end_time = time.time()
    print("Total CPU Time: ", end_time - start_time)
    #
    result_df = pd.DataFrame(columns=['ozone_id', 'dzone_id', 'path', 'generalized_cost'])
    for pair in od_pairs:
        path, total_generalized_cost, total_cost, total_travel_time = return_dict[pair]
        path = ';'.join(str(node) for node in path) if path is not None else None
        result_df = result_df.append({'ozone_id': pair[0], 'dzone_id': pair[1], 'path': path,
                                      'generalized_cost': total_generalized_cost, 'cost': total_cost,
                                      'travel_time': total_travel_time}, ignore_index=True)

    result_df.to_csv('auto_network/result.csv', index=False)
    result_df = pd.read_csv('auto_network/result.csv')
    ozone_result_df = result_df.groupby('ozone_id').mean()
    ozone_geometry_dict = dict(zip(node_df['node_id'], node_df['geometry']))
    ozone_result_df['geometry'] = [ozone_geometry_dict[ozone_id] for ozone_id in ozone_result_df.index]
    ozone_result_df['ozone_id'] = ozone_result_df.index
    ozone_result_df = ozone_result_df[['ozone_id', 'geometry', 'generalized_cost', 'cost', 'travel_time']]
    ozone_result_df.to_csv('auto_network/ozone_result.csv', index=False)

    # select 0.01% of demand pairs in result_df
    result_df = result_df.sample(frac=0.05)
    print("Number of demand pairs: ", len(result_df), "for visualization")
    # ozone_id, dzone_id, path
    vis_path_list = []
    for _, row in result_df.iterrows():
        path_list = row['path'].split(';')
        for i in range(len(path_list) - 1):
            pair = (row['ozone_id'], row['dzone_id'])
            from_node = int(path_list[i])
            to_node = int(path_list[i + 1])
            x_coord = node_df[node_df['node_id'] == from_node]['x_coord'].values[0]
            y_coord = node_df[node_df['node_id'] == from_node]['y_coord'].values[0]
            x_coord1 = node_df[node_df['node_id'] == to_node]['x_coord'].values[0]
            y_coord1 = node_df[node_df['node_id'] == to_node]['y_coord'].values[0]
            geometry = "LINESTRING (" + str(x_coord) + \
                       " " + str(y_coord) + ", " + str(x_coord1) + " " + str(y_coord1) + ")"
            # attach the VDF_fftt1 and VDF_penalty1 to the link
            fftt = link_df[(link_df['from_node_id'] == from_node) &
                           (link_df['to_node_id'] == to_node)]['VDF_fftt1'].values[0]
            cost = link_df[(link_df['from_node_id'] == from_node) &
                           (link_df['to_node_id'] == to_node)]['cost'].values[0]
            link_type_name = link_df[(link_df['from_node_id'] == from_node) &
                                     (link_df['to_node_id'] == to_node)]['link_type_name'].values[0]
            vis_path_list.append(
                [pair, from_node, to_node, geometry, fftt, cost, link_type_name])
    vis_path_df = pd.DataFrame(vis_path_list, columns=['pair', 'from_node', 'to_node', 'geometry', 'fftt',
                                                       'cost', 'link_type_name'])
    vis_path_df.to_csv('auto_network/vis_path.csv', index=False)
