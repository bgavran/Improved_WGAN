import os
import scipy.misc
import numpy as np


class ProjectPath:
    # base = json.loads(open("config.json").read()).get("path", "")
    # base = json.loads(open("config.json").read())["path"]
    base = os.path.dirname(os.path.dirname(__file__))

    def __init__(self, logdir):
        self.logdir = logdir

        from time import localtime, strftime
        self.timestamp = strftime("%B_%d__%H:%M", localtime())

        self.path = os.path.join(ProjectPath.base, self.logdir, self.timestamp)


