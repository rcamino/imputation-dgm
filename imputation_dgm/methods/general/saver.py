import time

import torch

from imputation_dgm.commandline import DelayedKeyboardInterrupt


class Saver(object):
    
    def __init__(self, path_by_models, logger, max_seconds_without_save):
        self.path_by_models = path_by_models
        self.logger = logger
        self.max_seconds_without_save = max_seconds_without_save
        
        self.last_flush_time = None
        self.kept_parameters = None
    
    def delayed_save(self, keep_parameters=False):
        now = time.time()

        # if this is the first save the time from last save is zero
        if self.last_flush_time is None:
            self.last_flush_time = now
            seconds_without_save = 0

        # if not calculate the time from last save
        else:
            seconds_without_save = now - self.last_flush_time

        # if too much time passed from last save
        if seconds_without_save > self.max_seconds_without_save:
            # save the current parameters
            self.save(only_use_kept=False)
            self.last_flush_time = now
            self.kept_parameters = None

        # if not too much time passed but parameters should be kept
        elif keep_parameters:
            self.kept_parameters = []
            for model, model_path in self.path_by_models.items():
                self.kept_parameters.append((model.state_dict(), model_path))

    def save(self, only_use_kept=False):
        with DelayedKeyboardInterrupt():
            # if kept parameters should be ignored the current model parameters are used
            if not only_use_kept:
                for model, model_path in self.path_by_models.items():
                    torch.save(model.state_dict(), model_path)

            # if kept parameters should be used and they are defined
            elif self.kept_parameters is not None:
                for parameters, model_path in self.kept_parameters:
                    torch.save(parameters, model_path)

            self.logger.flush()
