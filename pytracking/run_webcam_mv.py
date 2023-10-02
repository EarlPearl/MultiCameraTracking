import os
import sys
import argparse


env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import TrackerMV

def list_of_ints(arg):
    return list(map(int, arg.split(',')))
def main():

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

    tracker = TrackerMV(args.tracker_name, args.tracker_param)

    tracker.run_video_generic_mv(debug=args.debug, visdom_info=visdom_info, web_cam_ids=args.webcam_ids)


if __name__ == '__main__':
    main()
