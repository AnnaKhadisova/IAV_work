# 70k images for training, 10k images for validation
# dataset is taken from official website BDD100K
# useful classes: CURBS(maybe), drivable area; lane/single white (we can also have double white and dashed and solid); lane/crosswalk
# + cityscapes dataset for people, cars, fences  (this dataset is from german streets)
# kitti  : it is only in karlsruhe; i can use it for additional measurements

# my pipeline: 1) bdd100k + cityscapes for fences + tunnels from roboflow  2) cityscapes for everything 3) experiment with losses
# trial: 1) bdd100k for everything except fences

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw

img_size = (1280, 720)

folder_path = "D:\TwinLiteNetPlus-main\convert_jsons_into_pngmasks\json_files"
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)  # file path for one json file
    #print(file_path)
    with open(file_path, "r",  encoding="latin-1") as f:
        data = json.load(f)
    
    mask_da = Image.new("L", img_size, 0)   # mask for drivable area
    draw_da = ImageDraw.Draw(mask_da)

    mask_lm = Image.new("L", img_size, 0)   # mask for lane markings
    draw_lm = ImageDraw.Draw(mask_lm)

    mask_crossw = Image.new("L", img_size, 0)   # mask for lane markings
    draw_crossw = ImageDraw.Draw(mask_crossw)
   
    for frame in data["frames"]:
        for obj in frame["objects"]:
            if obj["category"] == "area/drivable":  # we are in a one image in a one drivable area
                    #print(obj["category"])
                    #print(obj["poly2d"])
                    
                pts = [tuple(map(int, v[:2])) for v in obj["poly2d"]]
                draw_da.polygon(pts, outline=1, fill=1)
                #print(pts)
            if obj["category"] == "lane/double white" or obj["category"] == "lane/double yellow" or obj["category"] == "lane/double other" or obj["category"] == "lane/single other" or obj["category"] == "lane/single white" or obj["category"] == "lane/single yellow":
                pts = [tuple(map(int, v[:2])) for v in obj["poly2d"]]
                draw_lm.polygon(pts, outline=1, fill=1)

            if obj["category"] == "lane/crosswalk":
                pts = [tuple(map(int, v[:2])) for v in obj["poly2d"]]
                draw_crossw.polygon(pts, outline=1, fill=1)

        mask_da.save(f"../bdd100k/drivable_area_annotations/train/{file_name.replace('json', 'png')}")
        mask_lm.save(f"../bdd100k/lane_line_annotations/train/{file_name.replace('json', 'png')}")
        mask_crossw.save(f"../bdd100k/crosswalks/train/{file_name.replace('json', 'png')}")



  

            




                    






#  in the end we need to save the mask into val!