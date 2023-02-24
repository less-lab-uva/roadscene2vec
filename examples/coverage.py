import networkx as nx
import json
from itertools import combinations
import copy
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import numpy as np

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


def computeSGCoverage(edit_thresholds):
    start_time = time.time()
    unique_count_dict = {}
    for edit_thresh in edit_thresholds:
        unique_counts = []
        f = open('output/' + str(edit_thresh)+".txt", 'w')

        print("Running with thresh:", edit_thresh)

        with open('enc_sgs.json','r') as fp:
            sgs = json.load(fp)

        print("Preprocessing Videos...")
        # # Step1: Extract unique scenegraphs from each of the videos
        videos = {0:{}}
        cur_vid = 0
        sg_ids = list(sgs.keys())

        # Preprocess all videos by removing frames with equivalent scenegraphs
        for i in tqdm(list(range(len(sg_ids)))):
            # convert SG encoding to SG
            sg_enc = sgs[sg_ids[i]]
            sg = nx.from_dict_of_dicts(sg_enc, create_using=nx.MultiDiGraph, multigraph_input = True)
            for node in sg.nodes():
                sg.nodes[node]['type'] = node.split("_")[0]
            flag = True

            # find all unique frames in each video
            for seen_sgname in videos[cur_vid]:

                seen_sg = nx.from_dict_of_dicts(videos[cur_vid][seen_sgname], create_using=nx.MultiDiGraph, multigraph_input = True)
                for node in seen_sg.nodes():
                    seen_sg.nodes[node]['type'] = node.split("_")[0]

                # if the current frame has already been seen (val <= thresh), stop and don't include it as a unique one
                val = nx.graph_edit_distance(sg, seen_sg, node_subst_cost = nsubst_cost, node_del_cost = ndel_cost, node_ins_cost = nins_cost, edge_subst_cost = esubst_cost, edge_del_cost = edel_cost, edge_ins_cost = eins_cost, upper_bound=edit_thresh, timeout = 0.01)
                # If an edit distance below the threshold exists, then consider it to be an equivalent graph (don't add it)
                if val != None:
                    flag = False
                    break

            # If the graph doesn't match with any others in the same video, then add it to unique set
            if flag:
                videos[cur_vid][i] = sg_enc

            # TODO: Currently assumes each video has 40 frames, but some have 39 or 41
            if i != 0 and i % 39 == 0:
                cur_vid +=1
                videos[cur_vid] = {}


        print("Finding smallest subset of videos... ")

        selected_vids = [] # video IDs selected so far
        unique = [] # SGs that are deemed unique
        prev_unique = [] # SGs of the last selected video
        loop_condition = True

        # Loop until all videos have been selected or full coverage
        while loop_condition:
            # for each video that has not yet been selected...
            for v in list(videos):
                if v not in selected_vids:
                    # ... iterate through its graphs and prune those that are not unique
                    for sg1name in list(videos[v]):
                        sg1 = nx.from_dict_of_dicts(videos[v][sg1name], create_using=nx.MultiDiGraph, multigraph_input = True)

                        # Set node type attribute; would be better to set this when generating SGs instead
                        for node in sg1.nodes():
                            sg1.nodes[node]['type'] = node.split("_")[0]

                        # for each SG in the previous selected video..
                        for sg2dict in prev_unique:
                            sg2 = nx.from_dict_of_dicts(prev_unique[sg2dict], create_using=nx.MultiDiGraph, multigraph_input = True)
                            for node in sg2.nodes():
                                sg2.nodes[node]['type'] = node.split("_")[0]

                            # compute edit distance between SGs; returns None if distance exceeds threshold
                            dist = nx.graph_edit_distance(sg1, sg2, node_subst_cost = nsubst_cost, node_del_cost = ndel_cost, node_ins_cost = nins_cost, edge_subst_cost = esubst_cost, edge_del_cost = edel_cost, edge_ins_cost = eins_cost, upper_bound = edit_thresh, timeout = 0.01)
                            
                            # if distance meets threshold, the SGs are unique and therefore videos[v][sg1name] was covered by the last video. Delete it!
                            if dist != None:
                                del videos[v][sg1name]
                                break
            
            # Of these candidate videos, select the one with the most unique videos remaining
            longest_len = max(map(len, videos.values()))
            max_lens = [k for k, v in videos.items() if len(v) == longest_len]
            best = max_lens[0]

            # if the best selected video has unique SGs
            if len(videos[best]) != 0:
                # Append all of its SGs to the set of all unique SGs
                for sg in videos[best]:
                    unique.append(videos[best][sg])
                prev_unique = copy.deepcopy(videos[best])
                selected_vids.append(best)
                del videos[best]
            else:
                loop_condition = False
                break
            
            # if there are no more candidate videos... stop
            if len(videos) == 0:
                loop_condition = False
                break
            
            f.write(str(len(selected_vids)) + " " + str(len(unique)) + " " + str(selected_vids) + "\n")
            unique_counts.append(len(unique))

        print("Total # of Unique SGs: ", len(unique))
        print("Total # of Videos Selected: ", len(selected_vids))
        print("Selected Videos: ", selected_vids)
        f.close()

        unique_count_dict[edit_thresh] = unique_counts

    print("Time Elapsed:", time.time() - start_time)


    ## PLOT RESULTS

    max_len = 0
    for d in unique_count_dict:
        if len(unique_count_dict[d]) > max_len:
            max_len = len(unique_count_dict[d])
    
    x = list(range(0,max_len))
    for d in unique_count_dict:
        while len(unique_count_dict[d]) < max_len:
            unique_count_dict[d].append(unique_count_dict[d][-1])
        plt.plot(x, unique_count_dict[d], label = 'd=' + str(d))

    plt.legend()
    plt.xticks(list(range(0,max_len,5)))
    plt.xlabel("# Videos Selected")
    plt.ylabel("# Unique Scenes Covered")
    plt.show()

    x = list(range(0,max_len))
    for d in unique_count_dict:
        plt.plot(x, np.array(unique_count_dict[d]) / unique_count_dict[d][-1], label = 'd=' + str(d))

    plt.legend()
    plt.xticks(list(range(0,max_len,5)))
    plt.xlabel("# Videos Selected")
    plt.ylabel("% Unique Scenes Covered")
    plt.show()


if __name__ == '__main__':
    edit_cost_threshs = [3000, 2000, 1000, 20, 10, 5, 3, 2, 1]
    computeSGCoverage(edit_cost_threshs)