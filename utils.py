
['car',
 'van',
 'Truck',
 'Pedestrian',
 'Person_sitting',
 'Cyclist',
 'Tram',
 'Misc',
 'DontCare']

objects = []
with open(label_file) as file:
    lines = file.readlines()
    for line in lines:
        items = line.split(" ")
        object_str = {'node_name': 'object',
                      'single_node_dict': {'name': items[0], 'pose': 'left', 'truncated': items[1], 'difficult': 0},
                      'child_node_name': 'bndbox',
                      'child_node_dict': {'xmin': items[4], 'ymin': items[5], 'xmax': items[6], 'ymax': items[7]}}
        objects.append(object_str)