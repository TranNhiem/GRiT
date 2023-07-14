'''\
TranNhiem 07/14/2023 
This code Features for Image Dense captioning Information 
    
'''

import argparse
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import sys
from itertools import islice

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

# sys.path.insert(0, 'third_party/CenterNet2/projects/CenterNet2/')

sys.path.insert(0, '/data/rick/autonomous_instruction_dataengine/Autonomous_Driving_Visual_Instruction_DataEngine/vision_language_model/GRiT/third_party/CenterNet2/projects/CenterNet2')

from centernet.config import add_centernet_config
sys.path.append( '/data/rick/autonomous_instruction_dataengine/Autonomous_Driving_Visual_Instruction_DataEngine/vision_language_model/GRiT/')

from grit.config import add_grit_config
from grit.predictor import VisualizationDemo

from detectron2.config import CfgNode as CN
import torch

# constants
WINDOW_NAME = "GRiT"


def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    if args.test_task:
        cfg.MODEL.TEST_TASK = args.test_task
    cfg.MODEL.BEAM_SIZE = 1
    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    cfg.USE_ACT_CHECKPOINT = False
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="/data/rick/autonomous_instruction_dataengine/Autonomous_Driving_Visual_Instruction_DataEngine/vision_language_model/GRiT/configs/GRiT_B_DenseCap_ObjectDet.yaml",
        # metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument(
        "--input",
        #default="/data/rick/autonomous_instruction_dataengine/Autonomous_Driving_Visual_Instruction_DataEngine/vision_language_model/GRiT/demo_images",
        default="/data1/dataset/Cityscapes/leftImg8bit/train/jena/", 
        #nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="/data/rick/autonomous_instruction_dataengine/Autonomous_Driving_Visual_Instruction_DataEngine/cityscape_test_imgs/grit_model_test",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--test-task",
        type=str,
        default='DenseCap',
        help="Choose a task to have GRiT perform",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[], # MODEL.WEIGHTS /data/rick/pretrained_weights/grit/grit_b_densecap_objectdet.pth
        nargs=argparse.REMAINDER,
    )
    return parser

def dense_pred_to_caption(predictions): 
    boxes = predictions["instances"].pred_boxes if predictions["instances"].has("pred_boxes") else None
    object_description = predictions["instances"].pred_object_descriptions.data
    new_caption = ""
    for i in range(len(object_description)):
        new_caption += (object_description[i] + ": " + str([int(a) for a in boxes[i].tensor.cpu().detach().numpy()[0]])) + "; "
    return new_caption

def resize_long_edge_cv2(image, target_size=384):
    height, width = image.shape[:2]
    aspect_ratio = float(width) / float(height)

    if height > width:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
    else:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image


def setup_cfg(args):
    cfg = get_cfg()
    if args["cpu"]:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file(args["config_file"])
    cfg.merge_from_list(args["opts"])
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args["confidence_threshold"]
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args["confidence_threshold"]
    if args["test_task"]:
        cfg.MODEL.TEST_TASK = args["test_task"]
    cfg.MODEL.BEAM_SIZE = 1
    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    cfg.USE_ACT_CHECKPOINT = False
    cfg.freeze()
    return cfg


def get_parser_(device):
    arg_dict = {'config_file': "/data/rick/autonomous_instruction_dataengine/Autonomous_Driving_Visual_Instruction_DataEngine/vision_language_model/GRiT/configs/GRiT_B_DenseCap_ObjectDet.yaml", 'cpu': False, 'confidence_threshold': 0.6, 'test_task': 'DenseCap', 'opts': ["MODEL.WEIGHTS", "/data/rick/pretrained_weights/grit/grit_b_densecap_objectdet.pth"]}
    if device == "cpu":
        arg_dict["cpu"] = True
    return arg_dict

def image_caption_api(image_src, device):
    args2 = get_parser_(device)
    cfg = setup_cfg(args2)
    demo = VisualizationDemo(cfg)
    if image_src:
        img = read_image(image_src, format="BGR")
        img = resize_long_edge_cv2(img, 384)
        print("this is image shape", img.shape)
        predictions, visualized_output = demo.run_on_image(img)
        new_caption = dense_pred_to_caption(predictions)
    return new_caption, predictions, visualized_output

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    demo = VisualizationDemo(cfg, device=device)

    if args.input:
        image_dir = args.input  # Directory containing the images
        num_images = 10  # Number of images to select
        image_paths = [os.path.join(args.input, filename) for filename in os.listdir(image_dir)]
        # Select the first 10 images using islice
        if len(image_paths) < num_images:
           num_images = len(image_paths)
        selected_image_paths = islice(image_paths, num_images)

        for path in tqdm.tqdm(selected_image_paths, disable=not args.output):
            # Get a list of all image paths in the directory

            #breakpoint()
            img = read_image(os.path.join(args.input, path), format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)

            new_caption = dense_pred_to_caption(predictions)
            breakpoint()
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                if not os.path.exists(args.output):
                    os.mkdir(args.output)
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit