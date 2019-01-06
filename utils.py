from enum import Enum

class BriscolaLogger:

    class LoggerLevels(Enum):
        DEBUG = 0
        PVP = 1
        TRAIN = 2
        TEST = 3


    def __init__(self, verbosity=None):
        if not verbosity:
            verbosity = self.LoggerLevels.TEST
        self.configure_logger(verbosity)


    def configure_logger(self, verbosity):

        if verbosity.value > self.LoggerLevels.DEBUG.value:
            self.DEBUG = lambda *args: None
        else:
            self.DEBUG = print

        if verbosity.value > self.LoggerLevels.PVP.value:
            self.PVP = lambda *args: None
        else:
            self.PVP = print

        if verbosity.value > self.LoggerLevels.TRAIN.value:
            self.TRAIN = lambda *args: None
        else:
            self.TRAIN = print

        self.TEST = print
