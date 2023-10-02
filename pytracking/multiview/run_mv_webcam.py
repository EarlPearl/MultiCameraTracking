import os
import sys
import argparse
from multiprocessing import Process, Queue, Manager
import cv2 as cv
from pytracking.utils.visdom import Visdom
from pytracking.evaluation.multi_object_wrapper import MultiObjectWrapper
import numpy as np
from collections import OrderedDict
from pytracking.utils.plotting import draw_figure, overlay_mask
from pathlib import Path




env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from tracker import Tracker

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

_tracker_disp_colors = {1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0),
                        4: (255, 255, 255), 5: (0, 0, 0), 6: (0, 255, 128),
                        7: (123, 123, 123), 8: (255, 128, 0), 9: (128, 0, 255)}

def multi_view_multi_single_tracking():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--use_visdom', type=bool, default=True, help='Flag to enable visdom')
    parser.add_argument('--visdom_server', type=str, default='127.0.0.1', help='Server for visdom')
    parser.add_argument('--visdom_port', type=int, default=8097, help='Port for visdom')
    parser.add_argument('--webcam_ids', type=list_of_ints, help='Webcam IDs')

    args = parser.parse_args()

    visdom_info = {'use_visdom': args.use_visdom, 'server': args.visdom_server, 'port': args.visdom_port}

    tracker = Tracker(args.tracker_name, args.tracker_param)

    web_cam_ids = args.webcam_ids
    queue = Queue(maxsize=1000)
    mp = list()
    for id in web_cam_ids:
        p = Process(target=tracker.run_video_generic, args=(queue,),
                    kwargs={"debug": args.debug, "visdom_info": visdom_info, "web_cam_id": id},
                    daemon=True)
        p.start()
        mp.append(p)

    while True:
        if not queue.empty():
            print("hei")
            item = queue.get()
            print("Object id: {}, state: {}".format(item[0], item[1]))
        if not all(p.is_alive() for p in mp):
            break
    for p in mp:
        p.join()
    print("All processes finished")



if __name__ == '__main__':
    multi_view_multi_single_tracking()
