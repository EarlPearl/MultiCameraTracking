import os
import sys
import argparse

env_path = os.path.join(os.path.dirname(__file__), '../..')
if env_path not in sys.path:
    sys.path.append(env_path)

from multi_camera_tracker import Tracker

def list_of_strings(arg):
    return list(map(str, arg.split(':')))

def multi_view_tracking_multi_process():
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """

    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('videofiles', type=list_of_strings, help='path to a video file.')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.set_defaults(save_results=False)

    args = parser.parse_args()

    tracker = Tracker(args.tracker_name, args.tracker_param)
    tracker.run_video_generic_mv(videofilepaths=args.videofiles, optional_box=args.optional_box, debug=args.debug, save_results=args.save_results)

def multi_view_tracking():
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """

    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')
    parser.add_argument('videofiles', type=list_of_strings, help='path to a video file.')
    parser.add_argument('--optional_box', type=float, default=None, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.set_defaults(save_results=False)

    args = parser.parse_args()

    tracker = Tracker(args.tracker_name, args.tracker_param)
    tracker.run_videofile_multiview(frame_count=1000,videofilepaths=args.videofiles, optional_box=args.optional_box, debug=args.debug, save_results=args.save_results)

def multi_camera_dataset():
    parser = argparse.ArgumentParser(description='Evaluate annoated dataset with video')
    parser.add_argument('folder_path', type=str, help='Path to root folder of dataset')
    args = parser.parse_args()
    tracker = Tracker()

    tracker.run_mc_dataset(args.folder_path)



if __name__ == '__main__':
    multi_camera_dataset()
