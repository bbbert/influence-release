import os
import datetime

class Logger(object):
    @property
    def log_path(self):
        return os.path.join(self.base_path, self.name)

    def __init__(self, base_path, name):
        self.base_path = base_path
        now = datetime.datetime.now()
        self.name = name + "_{}_{}.log".format(now.date, now.time)
    
    def log(self, message)
        with open(self.log_path, 'w') as f:
            f.write(message)
