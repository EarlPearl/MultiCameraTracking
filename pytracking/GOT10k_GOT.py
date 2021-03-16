from got10k.trackers import Tracker as GOT_Tracker
from got10k.experiments import ExperimentGOT10k
import numpy as np
import os
import sys
import argparse
import importlib
import pdb

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Sequence, Tracker

parser = argparse.ArgumentParser(description='Run GOT10K.')
parser.add_argument('--tracker_name', type=str, default='trdimp')
parser.add_argument('--tracker_param', type=str, default='trdimp') # trsiam / trdimp
parser.add_argument('--run_id', type=int, default=None)
parser.add_argument('--debug', type=int, default=0, help='Debug level.')
args = parser.parse_args()

TrTracker = Tracker(args.tracker_name, args.tracker_param, args.run_id)

class GOT_Tracker(GOT_Tracker):
    def __init__(self):
        super(GOT_Tracker, self).__init__(name='GOT_Tracker')
        self.tracker = TrTracker.tracker_class(TrTracker.get_parameters())

    def init(self, image, box):
        image = np.array(image)
        self.tracker.initialize(image, box)

    def update(self, image):
        image = np.array(image)
        self.box = self.tracker.track(image)
        return self.box


if __name__ == '__main__':
    # setup tracker
    tracker = GOT_Tracker()

    # run experiments on GOT-10k (validation subset)
    experiment = ExperimentGOT10k('/data1/wangning/GOT-10k', subset='test') ##### test val
    experiment.run(tracker, visualize=False)

    # report performance
    experiment.report([tracker.name])
