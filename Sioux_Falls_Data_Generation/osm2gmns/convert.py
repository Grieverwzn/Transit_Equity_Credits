import osm2gmns as og

default_lanes_dict = {'motorway': 4, 'trunk': 3, 'primary': 3, 'secondary': 2, 'tertiary': 2,
                      'residential': 1, 'service': 1, 'cycleway': 1, 'footway': 1, 'track': 1,
                      'unclassified': 1, 'connector': 2}
default_speed_dict = {'motorway': 120, 'trunk': 100, 'primary': 80, 'secondary': 60, 'tertiary': 40,
                      'residential': 30, 'service': 30, 'cycleway': 5, 'footway': 5, 'track': 30,
                      'unclassified': 30, 'connector': 120}
default_capacity_dict = {'motorway': 2300, 'trunk': 2200, 'primary': 1800, 'secondary': 1600, 'tertiary': 1200,
                         'residential': 1000, 'service': 800, 'cycleway': 800, 'footway': 800, 'track': 800,
                         'unclassified': 800, 'connector': 9999}
# convert osm file to network
net = og.getNetFromFile('sioux falls.osm', network_types='auto',
                        default_lanes=True, default_speed=True, default_capacity=True)

og.consolidateComplexIntersections(net, auto_identify=True)
og.outputNetToCSV(net)
