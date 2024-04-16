"""
JARVIS-MoCap (https://jarvis-mocap.github.io/jarvis-docs)
Copyright (c) 2022 Timo Hueser.
https://github.com/JARVIS-MoCap/JARVIS-HybridNet
Licensed under GNU Lesser General Public License v2.1
"""

import os
import csv
import itertools
import numpy as np
import torch
import cv2
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
import time
from ruamel.yaml import YAML


from jarvis.utils.reprojection import ReprojectionTool, load_reprojection_tools
from jarvis.utils.reprojection import get_repro_tool
from jarvis.config.project_manager import ProjectManager
from jarvis.prediction.jarvis3D import JarvisPredictor3D


def predict3D(params):
    #Load project and config
    project = ProjectManager()
    if not project.load(params.project_name):
        print (f'{CLIColors.FAIL}Could not load project: {project_name}! '
                    f'Aborting....{CLIColors.ENDC}')
        return
    cfg = project.cfg

    jarvisPredictor = JarvisPredictor3D(cfg, params.weights_center_detect,
                params.weights_hybridnet, params.trt_mode)

    reproTool = get_repro_tool(cfg, params.dataset_name)

    # Check if output dir not assigned
    if params.output_dir == "":
        params.output_dir = os.path.join(project.parent_dir,
                    cfg.PROJECTS_ROOT_PATH, params.project_name,
                    'predictions', 'predictions3D',
                    f'Predictions_3D_{time.strftime("%Y%m%d-%H%M%S")}')
    # If it is assigned build path to output
    else:
        params.output_dir = os.path.join(
            params.output_dir,
            f'Predictions_3D_{time.strftime("%Y%m%d-%H%M%S")}'
        )

    os.makedirs(params.output_dir, exist_ok = True)
    create_info_file(params)

    #create openCV video read streams
    video_paths = get_video_paths(
                params.recording_path, reproTool)
    caps, img_size = create_video_reader(params, reproTool,
                video_paths)

    # CR EDIT: make the number of frames go only until the end of the SHORTEST CAP so we
    # don't have to truncate the vids manually
    frame_counts = [c.get(cv2.CAP_PROP_FRAME_COUNT) for c in caps]
    min_frame_count = frame_counts[0]
    if frame_counts.count(frame_counts[0]) != len(frame_counts):
        # Then there are varying frame lens
        min_frame_count = int(min(frame_counts))
        print("Varying frame lengths found only analyzing up to the shortest" \
               f" vid (n frames = {min_frame_count})")

    if (params.number_frames == -1):
        params.number_frames = min_frame_count - params.frame_start
    else:
        assert params.frame_start+params.number_frames \
                    <= min_frame_count, \
                    "make sure your selected segment is not " \
                    "longer that the total video!"

    csvpath = os.path.join(params.output_dir, f'data3D_{params.trial_num}.csv')
    print(f"\nCreating file output: {csvpath}\n")
    csvfile = open(csvpath, 'w', newline='')
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    #if keypoint names are defined, add header to csvs
    if (len(cfg.KEYPOINT_NAMES) == cfg.KEYPOINTDETECT.NUM_JOINTS):
        create_header(writer, cfg)

    imgs_orig = np.zeros((len(caps), img_size[1],
                img_size[0], 3)).astype(np.uint8)

    for frame_num in tqdm(range(params.number_frames)):
        #load a batch of images from all cameras in parallel using joblib
        Parallel(n_jobs=12, require='sharedmem')(delayed(read_images)
                    (cap, slice, imgs_orig) for slice, cap in enumerate(caps))
        imgs = torch.from_numpy(
                imgs_orig).cuda().float().permute(0,3,1,2)[:, [2, 1, 0]]/255.

        points3D_net, confidences = jarvisPredictor(imgs,
                    reproTool.cameraMatrices.cuda(),
                    reproTool.intrinsicMatrices.cuda(),
                    reproTool.distortionCoefficients.cuda())

        if points3D_net != None:
            row = []
            for point, conf in zip(points3D_net.squeeze(), confidences.squeeze().cpu().numpy()):
                row = row + point.tolist() + [conf]
            writer.writerow(row)

        else:
            row = []
            for i in range(cfg.KEYPOINTDETECT.NUM_JOINTS*4):
                row = row + ['NaN']
            writer.writerow(row)

        if params.progress_bar != None:
            params.progress_bar.progress(float(frame_num+1)
                        / float(params.number_frames))

    for cap in caps:
        cap.release()
    csvfile.close()


def create_video_reader(params, reproTool, video_paths):
    caps = []
    img_size = [0,0]
    for i, path in enumerate(video_paths):
        cap = cv2.VideoCapture(path)
        cap.set(1,params.frame_start)
        img_size_new = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        assert (img_size == [0,0] or img_size == img_size_new), \
                    "All videos need to have the same resolution"
        img_size = img_size_new
        caps.append(cap)

    return caps, img_size


def get_video_paths(recording_path, reproTool):

    '''
    Handle two different cases
    Case 1: recording_paths is a list of file paths to videos
    Case 2: recording_paths is a directory of videos (this was the original use)
    '''

    # Case 1:
    if isinstance(recording_path, list):
        for rp in recording_path:
            assert os.path.isfile(rp), f'recording path ({rp}) does not exist'
        videos = recording_path  # These are full paths

    # Case 2:
    elif isinstance(recording_path, str):
        assert os.path.isdir(recording_path), f'params.recording_paths ({recording_path}) does not exist'
        videos = [os.path.join(recording_path, f)
                   for f in os.listdir(recording_path) if f.endswith('.mp4')]  # These are full paths

    video_paths = []
    for i, camera in enumerate(reproTool.cameras):
        for video in videos:
            vid_name = os.path.basename(video)
            if camera == vid_name.split('.')[0]:
                video_paths.append(video)  ## video is the full path now
        assert (len(video_paths) == i+1), \
                    "Missing Recording for camera " + camera
    return video_paths


def read_images(cap, slice, imgs):
    ret, img = cap.read()
    imgs[slice] = img.astype(np.uint8)


def create_header(writer, cfg):
    joints = list(itertools.chain.from_iterable(itertools.repeat(x, 4)
                for x in cfg.KEYPOINT_NAMES))
    coords = ['x','y','z', 'confidence']*len(cfg.KEYPOINT_NAMES)
    writer.writerow(joints)
    writer.writerow(coords)


def create_info_file(params):
    with open(os.path.join(params.output_dir, 'info.yaml'), 'w') as file:
        yaml=YAML()
        yaml.dump({'recording_path': params.recording_path,
                    'dataset_name': params.dataset_name,
                    'frame_start': params.frame_start,
                    'number_frames': params.number_frames}, file)
