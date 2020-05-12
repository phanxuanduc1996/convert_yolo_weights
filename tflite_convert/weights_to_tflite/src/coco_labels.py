import os

#from definitions import ROOT_DIR
from utils.randomcolor import RandomColor


class COCOLabels:

    @staticmethod
    def all(data_dir, yolo_version):
        with open(os.path.join(data_dir, yolo_version + '.names')) as f:
            return [l.strip() for l in f.readlines()]

    @staticmethod
    def colors(data_dir, yolo_version):
        # NOTE: Fix a seed so that we get the same colors for the classes
        rc = RandomColor(seed=123)

        return [tuple([int(v) for v in rgb.replace('rgb', '').replace('(', '').replace(')', '').split(',')])
                for rgb in rc.generate(count=len(COCOLabels.all(data_dir, yolo_version)), format_='rgb')]
