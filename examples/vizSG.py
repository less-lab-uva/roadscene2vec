import os
import sys
from io import BytesIO
from pathlib import Path
from glob import glob
import json
from PIL import Image
import networkx as nx
from networkx.drawing import nx_agraph, nx_pydot
import matplotlib.pyplot as plt
import numpy as np
import subprocess

if len(sys.argv) > 1:
    num_images = int(sys.argv[1])
else:
    num_images = 1000

def draw_scenegraph_pydot(sg):
  A = nx_pydot.to_pydot(sg)
  img = A.create_png()
  return Image.open(BytesIO(img))

with open('enc_sgs.json','r') as fp:
    d = json.load(fp)

idxs = np.random.randint(0, len(d), len(d))
i = 0
count = 0
while count < num_images:
    idx = idxs[i]
    sg1 = nx.from_dict_of_dicts(d[list(d.keys())[idx]], create_using=nx.MultiDiGraph, multigraph_input = True)

    if len(sg1.nodes()) > 9:
        img1 = Image.open("./use_case_data/lanechange/22_lanechange/raw_images/" + str(list(d.keys())[idx]) + ".jpg")

        plt.subplot(1, 2, 1)
        sg_img = draw_scenegraph_pydot(sg1)
        plt.imshow(sg_img)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(img1)
        plt.axis('off')

        # plt.show()
        subprocess.run(["mkdir","-p","./sampleSGviz/"])
        plt.savefig("./sampleSGviz/" + str(list(d.keys())[idx]) + ".jpg", dpi = 1000, bbox_inches = 'tight', pad_inches = 0)
        count += 1
    i += 1
    
