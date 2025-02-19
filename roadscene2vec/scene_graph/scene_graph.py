import matplotlib

matplotlib.use("Agg")
import networkx as nx
import sys
from pathlib import Path
# sys.path.append(str(Path("../../")))
#from roadscene2vec.scene_graph.relation_extractor import Relations, ActorType, RELATION_COLORS 
from roadscene2vec.roadscene2vec.scene_graph.nodes import Node
#from roadscene2vec.scene_graph.nodes import Node
from networkx.drawing import nx_pydot
import pandas as pd
import torch
import math
from collections import defaultdict
import numpy as np
import copy
import cv2

'''Create scenegraph using raw Carla json frame data or raw image data'''
class SceneGraph:
    
    #graph can be initialized with a framedict containing raw Carla data to load all objects at once
    def __init__(self, relation_extractor, framedict= None, framenum=None, bounding_boxes = None, segmentation=None, bev = None, coco_class_names=None, platform='carla', config=None):
        #configure relation extraction settings
        self.relation_extractor = relation_extractor
        
        self.platform = platform
        self.config = config
        
        if self.platform == "carla":
            self.g = nx.MultiDiGraph() #initialize scenegraph as networkx graph
            self.road_node = Node("Root Road", {"name":"Root Road"}, "road", self.relation_extractor.actors.index("road"))
            self.add_node(self.road_node)   #adding the road as the root node
            self.parse_json(framedict) # processing json framedict
        elif self.platform == "image":

            self.g = nx.MultiDiGraph()  # initialize scenegraph as networkx graph
            # road and lane settings.
            self.road_node = Node("Root Road", {"name":"Root Road"}, "road", self.relation_extractor.actors.index("road"))
            self.add_node(self.road_node)   # adding the road as the root node
    
            # set ego location to middle-bottom of image.
            # set ego location to middle-bottom of image.
            self.ego_location = bev.get_projected_point(
                                    bev.params['width']/2, 
                                    bev.params['height'])
            
            self.ego_location = bev.apply_depth_estimation(
                                    self.ego_location[0], 
                                    self.ego_location[1])
            
            #import pdb; pdb.set_trace()
            self.egoNode = Node('ego car', {

                                       'location_x': self.ego_location[0], 
                                       'location_y': self.ego_location[1]}, 
                                       'ego_car', self.relation_extractor.actors.index("ego_car"))
    
            # add ego-vehicle to graph
            self.add_node(self.egoNode)
            
            # add middle, right, and left lanes to graph
            self.relation_extractor.extract_relative_lanes(self) 
    
            # convert bounding boxes to nodes and build relations.
            if len(bounding_boxes) == 2:
                bounding_boxes = bounding_boxes[0]

            boxes, labels, image_size, pred_masks = bounding_boxes
            seg = segmentation

            self.get_nodes_from_bboxes(bev, boxes, seg, labels, coco_class_names, pred_masks)
            self.relation_extractor.extract_semantic_relations(self)

    # [CM]: Super fast method for padding a bitmask
    def getPadding(self, arr, n_pad):
        # accumulated padding
        ret = np.zeros(arr.shape)
        # translate entire matrix in each direction
        for param in [(n_pad,1),(-n_pad,1),(n_pad,0),(-n_pad,0)]:
            rl = copy.deepcopy(arr)
            # incrementally increase pad size, combine results
            for i in range(1, n_pad + 1):
                # shift array elements in one direction
                rl = np.roll(rl, np.sign(param[0]), axis=param[1])
                # values roll to opposing side, so reset to false
                val = 0 if np.sign(param[0]) == 1 else -1
                if param[1] == 1:
                    rl[:,val] = False
                else:
                    rl[val,:] = False
                # compute all newly covered pixels from transformation
                # determines overlap and then excludes overlap from result
                elems = np.logical_xor(rl, np.logical_and(arr, rl))
                # accumulate new pixels
                ret = np.logical_or(ret, elems)
        return ret


    def get_seg_masks(self, seg_info):
        seg = seg_info[0].cpu().detach().numpy()
        info = seg_info[1]
        d = {}
        for item in info:
            if item['category_id'] not in d:
                d[item['category_id']] = [item['id']]
            else:
                d[item['category_id']].append(item['id'])
        masks = []
        for item in info:
            mask = np.where(seg == item['id'], True, False)
            masks.append(mask)
        return masks


    # [CM]: Given a mask, compute all neighboring classes in segmented space
    def get_seg_region(self, seg_info, mask):
        segdata = seg_info[1]
        seg_masks = self.get_seg_masks(seg_info)

        mask = mask.cpu().detach().numpy()
        mask = self.getPadding(mask, 5)

        # these are just detectron2 object/background classes
        things = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'things', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water', 'window-blind', 'window', 'tree', 'fence', 'ceiling', 'sky', 'cabinet', 'table', 'floor', 'pavement', 'mountain', 'grass', 'dirt', 'paper', 'food', 'building', 'rock', 'wall', 'rug']
        numlist = list(range(len(things)))
        stuff = ['things', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water', 'window-blind', 'window', 'tree', 'fence', 'ceiling', 'sky', 'cabinet', 'table', 'floor', 'pavement', 'mountain', 'grass', 'dirt', 'paper', 'food', 'building', 'rock', 'wall', 'rug']
        numlist2 = list(range(len(stuff)))

        labels1 = dict(zip(numlist, things))
        labels2 = dict(zip(numlist2, stuff))

        names = []
        for item in segdata:
            name = ""
            if item['isthing']:
                name = labels1[item['category_id']]
            else:
                name = labels2[item['category_id']]
            names.append(name)

        overlap_classes = []
        for seg, name in zip(seg_masks, names):
            res = np.sum(np.logical_and(seg, mask))
            if res:
                overlap_classes.append(name)
        return overlap_classes


    def get_nodes_from_bboxes(self, bev, boxes, seg, labels, coco_class_names, pred_masks):
        for idx, (box, label, mask) in enumerate(zip(boxes, labels, pred_masks)):
            box = box.cpu().numpy().tolist()
            class_name = coco_class_names[label]
            #import pdb; pdb.set_trace()
            attr = {'left': box[0], 'top': box[1], 'right': box[2], 'bottom': box[3]}
            
            # exclude vehicle dashboard
            if attr['top'] >= bev.params['height'] - self.config.image_settings['BOTTOM_CROPPED']: continue;
            

            # filter traffic participants
            actor_type = ""
            for actor_ in range(len(self.relation_extractor.actors)):
                if class_name == self.relation_extractor.actors[actor_]:
                    actor_type = self.relation_extractor.actors[actor_]
                    actor_value = actor_
                elif f"{self.relation_extractor.actors[actor_].upper()}_NAMES" in self.relation_extractor.conf.relation_extraction_settings:
                    if class_name in self.relation_extractor.conf.relation_extraction_settings[f"{self.relation_extractor.actors[actor_].upper()}_NAMES"]: #ie specific car name
                        actor_type = self.relation_extractor.actors[actor_]
                        actor_value = actor_
            if actor_type == "": #if actor's type not included in ACTOR_NAMES
                continue

            # map center-bottom of bounding box to warped image
            x_mid = (attr['right'] + attr['left']) / 2
            y_bottom = attr['bottom']
            x_bev, y_bev = bev.get_projected_point(x_mid, y_bottom)

            # approximate locations / distances in feet
            attr['location_x'], attr['location_y'] = bev.apply_depth_estimation(x_bev, y_bev)
            

            # due to bev warp, vehicles far from horizon get warped behind car, thus we will default them as far from vehcile
            if attr['location_y'] > self.egoNode.attr['location_y']:
                # should store this in a list dictating the filename of the scene
                # print('BEV warped to behind vehicle')
                attr['location_y'] = self.egoNode.attr['location_y'] - self.relation_extractor.proximity_rels[-1][1] #assuming the last proximity threshold will be the most vague

            attr['rel_location_x'] = attr['location_x'] - self.egoNode.attr['location_x']           # x position relative to ego (neg left, pos right)
            attr['rel_location_y'] = attr['location_y'] - self.egoNode.attr['location_y']           # y position relative to ego (neg vehicle ahead of ego)
            attr['distance_abs'] = math.sqrt(attr['rel_location_x']**2 + attr['rel_location_y']**2) # absolute distance from ego

            # [CM]: Get all neighboring object classes
            attr['seg_regions'] = self.get_seg_region(seg, mask)
            attr['mask'] = mask.cpu().numpy()

            # [CM]: Construct and add node to graph
            # TODO filter based on location/size of node? ex:
            # AREA_THRESH = 7500
            # if 'road' in attr['seg_regions'] and np.sum(mask.cpu().detach().numpy().astype(np.uint8)) > AREA_THRESH:
            node = Node('%s_%d' % (actor_type, idx), attr, actor_type, actor_value)
            self.add_node(node)  # change
            # add lane vehicle relations to graph
            self.relation_extractor.add_mapping_to_relative_lanes(self, node)

    def add_node(self, node):
        '''Add a single node to graph. node can be any hashable datatype including objects'''
        color = 'white'
        if 'ego' in node.name.lower():
            color = 'red'
        elif 'car' in node.name.lower():
            color = 'green'
        elif 'lane' in node.name.lower():
            color = 'yellow'
        elif 'root' in node.name.lower():
            color = 'magenta'
        self.g.add_node(node, attr=node.attr, label=node.name, style='filled', fillcolor=color)


# add all pair-wise relations between two nodes
    def add_relations(self, relations_list):
        #import pdb; pdb.set_trace()
        for relation in relations_list:
            self.add_relation(relation)
    

    # add a single pair-wise relation between two nodes
    def add_relation(self, relation):
        if relation != []:
            node1, edge, node2 = relation
            if node1 in self.g.nodes and node2 in self.g.nodes:
                self.g.add_edge(node1, node2, value=self.relation_extractor.rels.index(edge), label=edge, color=self.relation_extractor.relational_colors[edge]) #relations might need to be turned into objects not just remain strings
                
            else:
                raise NameError("One or both nodes in relation do not exist in graph. Relation: " + str(relation))
            

    #parses actor dict and adds nodes to graph. this can be used for all actor types.
    def add_actor_dict(self, key, actordict):
        for actor_id, attr in actordict.items():
            # filter actors behind ego 
            n = Node(None, None, attr['name'], None)   #using the actor key as the node name and the dict as its attributes.
            n.label, n.value = self.relation_extractor.get_actor_type(n)   
            n.attr = attr
            x1, y1 = math.cos(math.radians(self.egoNode.attr['rotation'][0])), math.sin(math.radians(self.egoNode.attr['rotation'][0]))
            x2, y2 = attr['location'][0] - self.egoNode.attr['location'][0], attr['location'][1] - self.egoNode.attr['location'][1]
            inner_product = x1*x2 + y1*y2
            length_product = math.sqrt(x1**2+y1**2) + math.sqrt(x2**2+y2**2)
            degree = math.degrees(math.acos(inner_product / length_product))
            
            if key == "sign":
              import pdb; pdb.set_trace()
            
            if (degree <=190 or degree >= 350):#TEST FOR CARLA #if degree <= 80 or (degree >=280 and degree <= 360):
                # if abs(self.egoNode.attr['lane_idx'] - attr['lane_idx']) <= 1 \
                # or ("invading_lane" in self.egoNode.attr and (2*self.egoNode.attr['invading_lane'] - self.egoNode.attr['orig_lane_idx']) == attr['lane_idx']):
                n.name = n.label.lower() + "_" + actor_id
                
                self.add_node(n)
                self.relation_extractor.add_mapping_to_relative_lanes(self, n)
            
            

    #add the contents of a whole framedict to the graph
    def parse_json(self, framedict):
        
#        self.egoNode = Node("ego:"+framedict['ego']['name'], framedict['ego'], 'CAR')    
        self.egoNode = Node('ego car', framedict['ego'], 'ego_car', self.relation_extractor.actors.index("ego_car"))
        self.add_node(self.egoNode) #change

        #rotating axes to align with ego. yaw axis is the primary rotation axis in vehicles
        self.ego_yaw = math.radians(self.egoNode.attr['rotation'][0])
        self.ego_cos_term = math.cos(self.ego_yaw)
        self.ego_sin_term = math.sin(self.ego_yaw)
        self.relation_extractor.extract_relative_lanes(self)

#         self.relation_extractor = RelationExtractor(self.egoNode) #see line 99
        for key, attrs in framedict.items():   
            if key == 'actors' or key == 'sign':
              self.add_actor_dict(key, attrs)
        self.relation_extractor.extract_semantic_relations(self)
        

    def visualize(self, filename=None):
        #import pdb;pdb.set_trace()
        A = nx_pydot.to_pydot(self.g)
        A.write_png(filename)

    
#==========================================================================================
# this is for creation of trainer input using carla data
#==========================================================================================
    
    def get_carla_node_embeddings(self, feature_list):
        rows = []
        labels=[]
        ego_attrs = None
        
        #extract ego attrs for creating relative features
        for node, data in self.g.nodes.items():
            if "ego" in str(node):
                ego_attrs = data['attr']
        if ego_attrs == None:
            raise NameError("Ego not found in scenegraph")
    
        #rotating axes to align with ego. yaw axis is the primary rotation axis in vehicles
        ego_yaw = math.radians(ego_attrs['rotation'][0])
        cos_term = math.cos(ego_yaw)
        sin_term = math.sin(ego_yaw)
    
        def rotate_coords(x, y): 
            new_x = (x*cos_term) + (y*sin_term)
            new_y = ((-x)*sin_term) + (y*cos_term)
            return new_x, new_y
            
        def get_carla_embedding(node, row):
            row['type_'+str(node.value)] = 1 #assign 1hot class label
            return row
        
        for idx, node in enumerate(self.g.nodes):
            d = defaultdict()
            row = get_carla_embedding(node, d)
            labels.append(node.value)
            rows.append(row)
            
        embedding = pd.DataFrame(data=rows, columns=feature_list)
        embedding = embedding.fillna(value=0) #fill in NaN with zeros
        embedding = torch.FloatTensor(embedding.values)
        
        return embedding
    
    
    def get_carla_edge_embeddings(self, node_name2idx):
        edge_index = []
        edge_attr = []
        for src, dst, edge in self.g.edges(data=True):
            edge_index.append((node_name2idx[src], node_name2idx[dst]))
            edge_attr.append(edge['value'])
    
        edge_index = torch.transpose(torch.LongTensor(edge_index), 0, 1)
        edge_attr  = torch.LongTensor(edge_attr)
        
        return edge_index, edge_attr
    
    #===================================================================
    
    # this is for creation of trainer input using image data 
    #===================================================================
    
    def get_real_image_node_embeddings(self, feature_list):
        rows = []
        labels = []
        ego_attrs = None

        # extract ego attrs for creating relative features
        for node, data in self.g.nodes.items():
            if "ego" in str(node).lower():
                ego_attrs = data['attr']

        if ego_attrs == None:
            raise NameError("Ego not found in scenegraph")

        def get_real_embedding(node, row):
            # for key in self.feature_list:
            #     if key in node.attr:
            #         row[key] = node.attr[key]
            row['type_'+str(node.value)] = 1  # assign 1hot class label
            return row

        for idx, node in enumerate(self.g.nodes):
            d = defaultdict()
            row = get_real_embedding(node, d)
            
            labels.append(node.value)
            rows.append(row)

        embedding = pd.DataFrame(data=rows, columns=feature_list)
        embedding = embedding.fillna(value=0)  # fill in NaN with zeros
        embedding = torch.FloatTensor(embedding.values)
        #import pdb; pdb.set_trace()
        return embedding

    def get_real_image_edge_embeddings(self, node_name2idx):
      edge_index = []
      edge_attr = []
      for src, dst, edge in self.g.edges(data=True):
          #import pdb; pdb.set_trace()
          edge_index.append((node_name2idx[src], node_name2idx[dst]))
          edge_attr.append(edge['value'])
  
      edge_index = torch.transpose(torch.LongTensor(edge_index), 0, 1)
      edge_attr = torch.LongTensor(edge_attr)
  
      return edge_index, edge_attr
    
    #==================================================================
    
    
