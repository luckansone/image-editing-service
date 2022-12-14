import math

class EarlyStopping:
    def __init__(self, min_diff):
        self.counter = 0
        self.max_counter = 6
        self.min_diff = min_diff
        
    def early_stop(self, prev_val, cur_val):
        if (math.fabs(prev_val - cur_val) < self.min_diff):
            self.counter = self.counter + 1
        if (self.counter == self.max_counter):
            return True
        return False