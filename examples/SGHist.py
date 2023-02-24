import matplotlib.pyplot as plt
import numpy as np
import json


import networkx as nx
import json
from itertools import combinations


# [CM]: SG EDIT COSTS
def nsubst_cost(n1, n2): # Node substitutions
    superclasses = [['car', 'bus', 'truck'], ['bicycle', 'motorcycle'],['person']]
    costs = [[0, 999, 999],[999, 0, 999],[999, 999, 0]]
    cost = 999

    # check superclasses
    idx1, idx2 = None, None
    for i in range(len(superclasses)):
        if n1['type'] in superclasses[i]:
            idx1 = i
        if n2['type'] in superclasses[i]:
            idx2 = i
   
    if idx1 != None and idx2 != None:
        cost = costs[idx1][idx2]

    if n1['type'] == n2['type']:
        cost = 0

    return cost

def ndel_cost(n1): # Node deletions
    return 999

def nins_cost(n1): # Node insertions
    return 999

def esubst_cost(e1_inp,e2_inp): # Edge substitutions
    e1, e2 = e1_inp['label'], e2_inp['label']

    regions_costs = [[0, 0.5, 1, 0.75],[0.5, 0, 0.5, 0.5],[1,0.5,0,0.5],[0.75,0.5,0.5,0]]
    regions = {'inDFrontOf':0, 'inSFrontOf':1, 'atDRearOf':2, 'atSRearOf':3}

    dists_costs = [[0, 0.5, 1],[0.5, 0, 0.5],[1, 0.5, 0]]
    dists = [['near_coll', 'super_near'],['very_near', 'near'],['visible']]

    lr_costs = [[0,1],[1,0]]
    lr = {'toLeftOf':0, 'toRightOf':1}

    cost = 1

    if e1 in regions and e2 in regions: # if a region relation
        cost = regions_costs[regions[e1]][regions[e2]]
    elif e1 in lr and e2 in lr: # if left/right relation
        cost = lr_costs[lr[e1]][lr[e2]]
    else: # distance relation
        idx1, idx2 = None, None
        for n in range(len(dists)):
            idx1 = n if e1 in dists[n] else None
            idx2 = n if e2 in dists[n] else None
        if idx1 != None and idx2 != None:
            cost = dists_costs[idx1][idx2]

    if e1 == e2:
        cost = 0

    return cost

# Edge deletions/insertions
def edel_cost(e1):
    return 1

def eins_cost(e1):
    return 1



with open('enc_sgs.json','r') as fp:
    d = json.load(fp)


generatePairs = False

if generatePairs:
    f = open('./use_case_data/SGpairs.txt', 'w')
    res = {}
    i = 0
    pairs = list(combinations(d.keys(), 2))
    for pair in pairs:
        g1 = nx.from_dict_of_dicts(d[pair[0]], create_using=nx.MultiDiGraph, multigraph_input = True)
        g2 = nx.from_dict_of_dicts(d[pair[1]], create_using=nx.MultiDiGraph, multigraph_input = True)
        if len(g1.nodes()) != len(g2.nodes()) or len(g1.edges()) != len(g2.edges()):
            continue
        for g in [g1, g2]:
            for node in g.nodes():
                g.nodes[node]['type'] = node.split("_")[0]

        dist = nx.graph_edit_distance(g1, g2, node_subst_cost = nsubst_cost, node_del_cost = ndel_cost, node_ins_cost = nins_cost, edge_subst_cost = esubst_cost, edge_del_cost = edel_cost, edge_ins_cost = eins_cost, upper_bound=0, timeout = 0.01)
        if dist != None and dist <= 10.0:
            print(pair[0] + "," + pair[1] + "," + str(dist))
            f.write(pair[0] + "," + pair[1] + "," + str(dist) + "\n")
    f.close()



unique = {}
seen = np.zeros(40000)

# This text file has all pairs of SGs that are considered to be equivalent
f = open("./use_case_data/SGpairs.txt", 'r')

for s in f.readlines():
    line = s.split(",")
    g1 = int(line[0])
    g2 = int(line[1])

    if seen[g1] == 0 and (g1 not in unique) and (g2 not in unique): # if neither has been covered, add g1 as representative SG
        unique[g1] = [g2]
    elif (g1 in unique): # if g1 is already a representative unique SG, add g2 to its list
        unique[g1].append(g2)
    elif (g2 in unique): # if g2 is already a representative unique SG, add g1 to its list
        unique[g2].append(g1)

    # mark all scenegraphs as seen, so we don't add it twice
    if seen[g1] == 0:
        seen[g1] = 1
    if seen[g2] == 0:
        seen[g2] = 1

f.close()

# this list contains the sorted ordering of the most "popular" scenegraph to the least popular
ordered = dict(sorted(unique.items(), key=lambda item: len(item[1]), reverse=True))

lengths = []

for x in ordered.values():
    lengths.append(len(x))

plt.title("Edit distance <= 0")
plt.xlabel("SceneGraph ID")
plt.ylabel("# of Instances")
plt.xlim((0,len(lengths)))
plt.ylim((0,1000))
plt.bar(list(range(len(lengths))), height = lengths)
plt.show()