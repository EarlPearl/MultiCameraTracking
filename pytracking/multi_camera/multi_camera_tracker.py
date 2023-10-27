import importlib
import os
import threading

import cv2
import numpy
import itertools
import numpy as np
from collections import OrderedDict
from pytracking.evaluation.environment import env_settings
import time
import cv2 as cv
from pytracking.utils.visdom import Visdom
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pytracking.utils.plotting import draw_figure, overlay_mask
from pytracking.utils.convert_vot_anno_to_rect import convert_vot_anno_to_rect
from ltr.data.bounding_box_utils import masks_to_bboxes
from pytracking.evaluation.multi_object_wrapper import MultiObjectWrapper
from pathlib import Path
import torch, torchvision
from pytracking.features import preprocessing
from multiprocessing import Process, Queue
from PIL import Image
import Siamese_network
import torchvision.transforms as transforms
import torch.nn.functional as F

_tracker_disp_colors = {1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0),
                        4: (255, 255, 255), 5: (0, 0, 0), 6: (0, 255, 128),
                        7: (123, 123, 123), 8: (255, 128, 0), 9: (128, 0, 255)}

_tracker_disp_global_colors = {1: (255, 0, 0), 2: (255, 128, 0), 3: (128, 0, 255),
                               4: (255, 255, 255), 5: (0, 0, 0), 6: (0, 255, 128)}

_tracker_global_id = {(1, 1): 1, (1, 2): 2, (1, 3): 3, (1, 4): 4, (2, 2): 5,
                      (2, 3): 6, (2, 4): 7, (3, 3): 8, (3, 4): 9, (4, 4): 10}



class Tracker:

    def __init__(self, sc_tracker: str = None, sc_parameter_name: str = None, run_id: int = None):
        assert run_id is None or isinstance(run_id, int)

        self.sc_tracker_name = sc_tracker
        self.sc_parameter_name = sc_parameter_name
        self.run_id = run_id

        env = env_settings()

        self.mc_tracker = Siamese_network()
        self.mc_tracker.load()

        if sc_tracker is not None:
            tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tracker',
                                                                  self.sc_tracker_name))
            if os.path.isdir(tracker_module_abspath):
                tracker_module = importlib.import_module('pytracking.multiview.{}'.format(self.sc_tracker_name))
                self.sc_tracker_class = tracker_module.get_tracker_class()
            else:
                self.sc_tracker_class = None

    def run_mc_dataset(self):


    def run_mc_video_live(self, debug=None, visdom_info=None, videofilepaths=None, optional_box=None,
                             save_results=False, web_cam_ids=None):
        queue = Queue(maxsize=1000)
        mp = list()

        if videofilepaths is None:  # Create multiple instances of webcam
            for id in web_cam_ids:
                p = Process(target=self.run_video_generic_sc, args=(queue,),
                            kwargs={"debug": debug, "visdom_info": visdom_info, "web_cam_id": id,
                                    "tracker_id": len(mp)},
                            daemon=True)
                p.start()
                mp.append(p)

        else:  # Create multiple instances of videofile
            for path in videofilepaths:
                p = Process(target=self.run_video_generic_sc, args=(queue,),
                            kwargs={"debug": debug, "videofilepath": path, "tracker_id": len(mp)},
                            daemon=True)
                p.start()
                mp.append(p)
        imageopen = False
        Images = OrderedDict
        Images = {i: list() for i in range(len(mp))}
        if len(mp) > 0:
            while True:
                if not queue.empty():
                    tracker_output = queue.get()
                    for obj_id, target_patch in tracker_output['target_patch'].items():
                        Images[tracker_output['tracker_id']].append(target_patch)
                        # if not imageopen:
                        #     image = preprocessing.torch_to_numpy(target_patch)
                        #     img_norm = image / 255
                        #     im_resize = cv.resize(img_norm, (1000, 600))
                        #     imageopen = True

                if not all(p.is_alive() for p in mp):
                    break
                # if imageopen:
                #    cv.imshow('bilde', im_resize)

                key = cv.waitKey(1)
                if key == ord('q'):
                    break

            cv.destroyAllWindows()
            for p in mp:
                p.join()
            print("All tracking processes finished")
            # transformation = transforms.Compose([transforms.Resize((100, 100)),
            #                                      transforms.ToTensor()
            #                                      ])
            transformation = transforms.Resize((100, 100), antialias=False)
            net = Siamese_network.SiameseNetwork(backbone='resnet18').cuda()
            net.load_model()
            images1 = Images[0]
            images2 = Images[1]
            for i in range(min(len(images1), len(images2), 5)):
                img1 = transformation(images1[i]) / 255
                img2 = transformation(images2[i]) / 255
                concatenated = torch.cat((img1, img2), 0)
                out1, out2 = net.forward(img1.cuda(), img2.cuda())
                euclidean_distance = F.pairwise_distance(out1, out2)
                print(euclidean_distance.item())
                image = np.transpose(torchvision.utils.make_grid(concatenated).numpy(), (1, 2, 0))
                cv.imshow(f'Dissimilarity: {euclidean_distance.item():.2f}', image)
            cv.waitKey()
            cv.destroyAllWindows()

    def run_mc_video_file(self, frame_count, debug=None, visdom_info=None, videofilepaths=None,
                                optional_box=None,
                                save_results=False):
        controls = self.read_tracks(frame_count, videofilepaths)
        controls = self.compare_tracks(controls, frame_count)

        for id, control in controls.items():
            control['display_name'] = "Tracker ID: {} ".format(control['ID'])

            cv.namedWindow(control['display_name'], cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
            cv.resizeWindow(control['display_name'], 960, 720)

            control['frame_number'] = 0

            assert os.path.isfile(control['videofilepath']), "Invalid param {}".format(control['videofilepath'])
            ", videofilepath must be a valid videofile"
            control['cap'] = cv.VideoCapture(control['videofilepath'])
            ret, frame = control['cap'].read()
            control['frame_number'] += 1
            cv.imshow(control['display_name'], frame)

        frame_number = 0
        while True:
            if frame_number >= frame_count:
                break
            for id, control in controls.items():

                ret, frame = control['cap'].read()
                if frame is None:
                    break
                frame_disp = frame.copy()
                if frame_number in control['tracker_outputs']:
                    out = control['tracker_outputs'][frame_number]
                    for obj_id, state in out['target_bbox'].items():
                        state = [int(s) for s in state]
                        if obj_id in out['id_map']:
                            global_id = out['id_map'][obj_id]
                            cv.rectangle(frame_disp, (state[0], state[1]),
                                         (state[2] + state[0], state[3] + state[1]),
                                         _tracker_disp_global_colors[global_id], 5)

                        else:
                            cv.rectangle(frame_disp, (state[0], state[1]),
                                         (state[2] + state[0], state[3] + state[1]),
                                         _tracker_disp_colors[obj_id], 5)

                cv.imshow(control['display_name'], frame_disp)

            frame_number += 1
            key = cv.waitKey(1)
            if key == ord('q'):
                break

        for id, control in controls.items():
            control['cap'].release()
        cv.destroyAllWindows()

    def get_global_id(self, par):
        par = tuple(sorted(par))
        return _tracker_global_id[par]

    def compare_tracks(self, controls, frame_count):
        transformation = transforms.Resize((100, 100), antialias=False)
        net = Siamese_network.SiameseNetwork(backbone='resnet18').cuda()
        net.load_model()
        threshold = 0.6
        for frame in range(frame_count):
            target_patches = dict()
            for id, control in controls.items():
                if frame in control['tracker_outputs']:
                    target_patches[id] = control['tracker_outputs'][frame]['target_patch']
                    control['tracker_outputs'][frame]['id_map'] = dict()
            if len(target_patches) >= 2:
                for i in range(len(target_patches)):
                    matches = dict()
                    for j in range(len(target_patches)):
                        if i == j:
                            continue
                        else:
                            for k in range(1, len(target_patches[i]) + 1):
                                scores = list()
                                best_pair = tuple()
                                best_score = 0
                                for l in range(1, len(target_patches[j]) + 1):
                                    img1 = transformation(target_patches[i][k]) / 255
                                    img2 = transformation(target_patches[j][l]) / 255
                                    out1, out2 = net.forward(img1.cuda(), img2.cuda())
                                    euclidean_distance = F.pairwise_distance(out1, out2).item()
                                    scores.append(euclidean_distance)
                                    print(euclidean_distance)
                                    if euclidean_distance <= min(scores):
                                        best_pair = (k, l)

                                matches[best_pair] = min(scores)
                    for par in matches:
                        print(matches[par])
                        if matches[par] <= threshold:
                            global_id = self.get_global_id(par)
                            controls[i]['tracker_outputs'][frame]['id_map'][par[0]] = global_id
        return controls

    def read_tracks(self, frame_count, videofilepaths, debug=None, visdom_info=None, optional_box=None,
                    save_results=False):
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        controls = dict()

        count = 0
        # Init UI controll and tracker for each videofile:
        for videofilepath in videofilepaths:

            control = dict()
            control['ID'] = count
            count += 1
            control['videofilepath'] = videofilepath

            if multiobj_mode == 'default':
                tracker = self.create_tracker(params)
                if hasattr(tracker, 'initialize_features'):
                    tracker.initialize_features()
            elif multiobj_mode == 'parallel':
                tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
            else:
                raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

            ui_control = UIControl()
            display_name = "Tracker: {}, ID: {} ".format(self.name, control['ID'])

            cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
            cv.resizeWindow(display_name, 960, 720)
            cv.setMouseCallback(display_name, ui_control.mouse_callback)

            frame_number = 0

            if videofilepath is not None:
                assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
                ", videofilepath must be a valid videofile"
                cap = cv.VideoCapture(videofilepath)
                ret, frame = cap.read()
                frame_number += 1
                cv.imshow(display_name, frame)

            next_object_id = 1
            sequence_object_ids = []
            prev_output = OrderedDict()
            output_boxes = OrderedDict()
            tracker_outputs = OrderedDict()

            if optional_box is not None:
                assert isinstance(optional_box, (list, tuple))
                assert len(optional_box) == 4, "valid box's format is [x,y,w,h]"

                out = tracker.initialize(frame, {'init_bbox': OrderedDict({next_object_id: optional_box}),
                                                 'init_object_ids': [next_object_id, ],
                                                 'object_ids': [next_object_id, ],
                                                 'sequence_object_ids': [next_object_id, ]})

                prev_output = OrderedDict(out)

                output_boxes[next_object_id] = [optional_box, ]
                sequence_object_ids.append(next_object_id)
                next_object_id += 1

            # Wait for initial bounding box if video!
            paused = videofilepath is not None

            while True:

                if frame_number >= frame_count:
                    break
                if not paused:
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    frame_number += 1
                    if frame is None:
                        break

                frame_disp = frame.copy()

                info = OrderedDict()
                info['previous_output'] = prev_output

                if ui_control.new_init:
                    ui_control.new_init = False
                    init_state = ui_control.get_bb()

                    info['init_object_ids'] = [next_object_id, ]
                    info['init_bbox'] = OrderedDict({next_object_id: init_state})
                    sequence_object_ids.append(next_object_id)

                    output_boxes[next_object_id] = [init_state, ]
                    next_object_id += 1

                # Draw box
                if ui_control.mode == 'select':
                    cv.rectangle(frame_disp, ui_control.get_tl(), ui_control.get_br(), (255, 0, 0), 2)

                if len(sequence_object_ids) > 0:
                    info['sequence_object_ids'] = sequence_object_ids
                    out = tracker.track(frame, info)
                    prev_output = OrderedDict(out)

                    if 'segmentation' in out:
                        frame_disp = overlay_mask(frame_disp, out['segmentation'])
                        mask_image = np.zeros(frame_disp.shape, dtype=frame_disp.dtype)

                        if save_results:
                            mask_image = overlay_mask(mask_image, out['segmentation'])
                            if not os.path.exists(self.results_dir):
                                os.makedirs(self.results_dir)
                            cv.imwrite(self.results_dir + f"seg_{frame_number}.jpg", mask_image)

                    if 'target_bbox' in out:
                        tracker_outputs[frame_number] = out
                        for obj_id, state in out['target_bbox'].items():
                            state = [int(s) for s in state]
                            cv.rectangle(frame_disp, (state[0], state[1]),
                                         (state[2] + state[0], state[3] + state[1]),
                                         _tracker_disp_colors[obj_id], 5)

                            output_boxes[obj_id].append(state)
                # Put text
                font_color = (255, 255, 255)
                msg = "Select target(s). Press 'r' to reset or 'q' to quit."
                cv.rectangle(frame_disp, (5, 5), (630, 40), (50, 50, 50), -1)
                cv.putText(frame_disp, msg, (10, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 2)
                if videofilepath is not None:
                    msg = "Press SPACE to pause/resume the video."
                    cv.rectangle(frame_disp, (5, 50), (530, 90), (50, 50, 50), -1)
                    cv.putText(frame_disp, msg, (10, 75), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 2)

                # Display the resulting frame
                cv.imshow(display_name, frame_disp)
                key = cv.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    next_object_id = 1
                    sequence_object_ids = []
                    prev_output = OrderedDict()

                    info = OrderedDict()

                    info['object_ids'] = []
                    info['init_object_ids'] = []
                    info['init_bbox'] = OrderedDict()
                    tracker.initialize(frame, info)
                    ui_control.mode = 'init'
                # 'Space' to pause video
                elif key == 32 and videofilepath is not None:
                    paused = not paused
            control['tracker_outputs'] = tracker_outputs
            controls[control['ID']] = control
            cap.release()
            cv.destroyAllWindows()

        # run video seqences with the collected tracking
        return controls

    def run_video_generic_sc(self, queue, tracker_id, debug=None, visdom_info=None, videofilepath=None,
                          optional_box=None,
                          save_results=False, web_cam_id=None):
        """Run the tracker with the webcam or a provided video file.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name

        self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)
            if hasattr(tracker, 'initialize_features'):
                tracker.initialize_features()
        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        ui_control = UIControl()
        display_name = 'Display: ' + self.name
        if web_cam_id is not None:
            display_name += " Webcam ID: {}".format(web_cam_id)
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        cv.setMouseCallback(display_name, ui_control.mouse_callback)

        frame_number = 0

        if videofilepath is not None:
            assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
            ", videofilepath must be a valid videofile"
            cap = cv.VideoCapture(videofilepath)
            ret, frame = cap.read()
            frame_number += 1
            cv.imshow(display_name, frame)
        else:
            if web_cam_id is None:
                cap = cv.VideoCapture(0)
            else:
                cap = cv.VideoCapture(web_cam_id)

        next_object_id = 1
        sequence_object_ids = []
        prev_output = OrderedDict()
        output_boxes = OrderedDict()

        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's format is [x,y,w,h]"

            out = tracker.initialize(frame, {'init_bbox': OrderedDict({next_object_id: optional_box}),
                                             'init_object_ids': [next_object_id, ],
                                             'object_ids': [next_object_id, ],
                                             'sequence_object_ids': [next_object_id, ]})

            prev_output = OrderedDict(out)

            output_boxes[next_object_id] = [optional_box, ]
            sequence_object_ids.append(next_object_id)
            next_object_id += 1

        # Wait for initial bounding box if video!
        paused = videofilepath is not None

        while True:

            if not paused:
                # Capture frame-by-frame
                ret, frame = cap.read()
                frame_number += 1
                if frame is None:
                    break

            frame_disp = frame.copy()

            info = OrderedDict()
            info['previous_output'] = prev_output

            if ui_control.new_init:
                ui_control.new_init = False
                init_state = ui_control.get_bb()

                info['init_object_ids'] = [next_object_id, ]
                info['init_bbox'] = OrderedDict({next_object_id: init_state})
                sequence_object_ids.append(next_object_id)
                if save_results:
                    output_boxes[next_object_id] = [init_state, ]
                next_object_id += 1

            # Draw box
            if ui_control.mode == 'select':
                cv.rectangle(frame_disp, ui_control.get_tl(), ui_control.get_br(), (255, 0, 0), 2)

            if len(sequence_object_ids) > 0:
                info['sequence_object_ids'] = sequence_object_ids
                out = tracker.track(frame, info)
                prev_output = OrderedDict(out)

                if 'segmentation' in out:
                    frame_disp = overlay_mask(frame_disp, out['segmentation'])
                    mask_image = np.zeros(frame_disp.shape, dtype=frame_disp.dtype)

                    if save_results:
                        mask_image = overlay_mask(mask_image, out['segmentation'])
                        if not os.path.exists(self.results_dir):
                            os.makedirs(self.results_dir)
                        cv.imwrite(self.results_dir + f"seg_{frame_number}.jpg", mask_image)

                if 'target_bbox' in out:
                    for obj_id, state in out['target_bbox'].items():
                        state = [int(s) for s in state]
                        cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                                     _tracker_disp_colors[obj_id], 5)
                        if save_results:
                            output_boxes[obj_id].append(state)
                    if not queue.full():
                        out['tracker_id'] = tracker_id
                        queue.put(out)
                    else:
                        print("Error: queue full")

            # Put text
            font_color = (255, 255, 255)
            msg = "Select target(s). Press 'r' to reset or 'q' to quit."
            cv.rectangle(frame_disp, (5, 5), (630, 40), (50, 50, 50), -1)
            cv.putText(frame_disp, msg, (10, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 2)
            if videofilepath is not None:
                msg = "Press SPACE to pause/resume the video."
                cv.rectangle(frame_disp, (5, 50), (530, 90), (50, 50, 50), -1)
                cv.putText(frame_disp, msg, (10, 75), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, font_color, 2)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                next_object_id = 1
                sequence_object_ids = []
                prev_output = OrderedDict()

                info = OrderedDict()

                info['object_ids'] = []
                info['init_object_ids'] = []
                info['init_bbox'] = OrderedDict()
                tracker.initialize(frame, info)
                ui_control.mode = 'init'
            # 'Space' to pause video
            elif key == 32 and videofilepath is not None:
                paused = not paused

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = "webcam" if videofilepath is None else Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))
            print(f"Save results to: {base_results_path}")
            for obj_id, bbox in output_boxes.items():
                tracked_bb = np.array(bbox).astype(int)
                bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')



class UIControl:
    def __init__(self):
        self.mode = 'init'  # init, select, track
        self.target_tl = (-1, -1)
        self.target_br = (-1, -1)
        self.new_init = False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and self.mode == 'init':
            self.target_tl = (x, y)
            self.target_br = (x, y)
            self.mode = 'select'
        elif event == cv.EVENT_MOUSEMOVE and self.mode == 'select':
            self.target_br = (x, y)
        elif event == cv.EVENT_LBUTTONDOWN and self.mode == 'select':
            self.target_br = (x, y)
            self.mode = 'init'
            self.new_init = True

    def get_tl(self):
        return self.target_tl if self.target_tl[0] < self.target_br[0] else self.target_br

    def get_br(self):
        return self.target_br if self.target_tl[0] < self.target_br[0] else self.target_tl

    def get_bb(self):
        tl = self.get_tl()
        br = self.get_br()

        bb = [min(tl[0], br[0]), min(tl[1], br[1]), abs(br[0] - tl[0]), abs(br[1] - tl[1])]
        return bb


class CVatAnnonationWrapper:
    def __init__(self, folder):
        self.root = folder




