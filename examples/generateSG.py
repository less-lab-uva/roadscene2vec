import os
import sys
import json
import networkx as nx
sys.path.append(os.path.dirname(sys.path[0]))
from roadscene2vec.util.config_parser import configuration
from roadscene2vec.scene_graph.extraction import image_extractor as RealEx
from roadscene2vec.scene_graph.extraction import carla_extractor as CarlaEx
from roadscene2vec.scene_graph.extraction.extractor import Extractor as ex
from roadscene2vec.scene_graph.scene_graph import SceneGraph
from roadscene2vec.scene_graph.extraction.image_extractor import RealExtractor
from roadscene2vec.data.dataset import RawImageDataset
from tqdm import tqdm
from pathlib import Path

# Utilities
def get_extractor(config):
  return RealExtractor(config)

def get_data(extractor):
  temp = RawImageDataset()
  temp.dataset_save_path = extractor.input_path
  return temp.load().data

def get_bev(extractor):
  return extractor.bev#.warpPerspective(frame)

def get_bbox(extractor, frame):
  return extractor.get_bounding_boxes(frame)

def get_scenegraph(extractor, bbox, seg, bev):
  scenegraph = SceneGraph(extractor.relation_extractor,   
                          bounding_boxes=bbox, 
                          segmentation = seg,
                          bev=bev,
                          coco_class_names=extractor.coco_class_names, 
                          platform=extractor.dataset_type)
  return scenegraph.g


def generateSceneGraphs(extraction_config):
  print("Generating SceneGraphs...")
  extractor = get_extractor(extraction_config)
  dataset_dir = extractor.conf.location_data["input_path"]
  if not os.path.exists(dataset_dir):
    raise FileNotFoundError(dataset_dir)
  all_sequence_dirs = [x for x in Path(dataset_dir).iterdir() if x.is_dir()]
  all_sequence_dirs = sorted(all_sequence_dirs, key=lambda x: int(x.stem.split('_')[0]))
  all_sg_dicts = {}

  # [CM]: Generate SceneGraphs for all images in the provided directory
  for path in all_sequence_dirs:
    sequence = extractor.load_images(path)
    for frame in tqdm(sorted(sequence.keys())):
      # [CM]: Get detections + BEV projection, then generate SG object
      bbox, seg = get_bbox(extractor, sequence[frame])
      bev = get_bev(extractor)
      sg = get_scenegraph(extractor, bbox, seg, bev)

      # [CM]: Convert to dictionary for storage
      sg_dict = nx.to_dict_of_dicts(sg)

      # [CM]: Replace node instances with their names to save space (there is certainly a much cleaner way of doing this...)
      for m in list(sg_dict):
        for n in list(sg_dict[m]):
          for o in list(sg_dict[m][n]):
            if type(o) != str and type(o) != int and type(o) != dict:
              sg_dict[m][n][o.name] = sg_dict[m][n][o]
              del sg_dict[m][n][o]
          if type(n) != str and type(n) != int and type(n) != dict:
            sg_dict[m][n.name] = sg_dict[m][n]
            del sg_dict[m][n]
        if type(m) != str and type(m) != int and type(m) != dict:
          sg_dict[m.name] = sg_dict[m]
          del sg_dict[m]
      all_sg_dicts[frame] = sg_dict

  # [CM]: Save SG encodings to JSON
  with open('enc_sgs.json','w') as fp:
    json.dump(all_sg_dicts, fp, indent=4)

if __name__ == "__main__":
  #create scenegraph extraction config object
  scenegraph_extraction_config = configuration(r"scenegraph_extraction_config.yaml",from_function = True)
  generateSceneGraphs(scenegraph_extraction_config)