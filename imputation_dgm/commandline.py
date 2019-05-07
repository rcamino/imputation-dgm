from __future__ import print_function

import os
import signal
import sys


def parse_int_list(comma_separated_ints, separator=","):
    if comma_separated_ints is None or comma_separated_ints == "":
        return []
    return [int(i) for i in comma_separated_ints.split(separator)]


def parse_dict(comma_separated_key_values, list_separator=",", key_value_separator=":", value_type=str):
    result = dict()
    if comma_separated_key_values is not None and comma_separated_key_values != "":
        for key_value in comma_separated_key_values.split(list_separator):
            key, value = key_value.split(key_value_separator)
            result[key] = value_type(value)
    return result


def create_directories_if_needed(directory_path):
    if not os.path.exists(directory_path):
        try:
            os.makedirs(directory_path)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise
    return directory_path


def create_parent_directories_if_needed(file_path):
    create_directories_if_needed(os.path.dirname(file_path))
    return file_path


class DelayedKeyboardInterrupt(object):

    SIGNALS = [signal.SIGINT, signal.SIGTERM]

    def __init__(self):
        self.signal_received = {}
        self.old_handler = {}

    def __enter__(self):
        self.signal_received = {}
        self.old_handler = {}
        for sig in self.SIGNALS:
            self.old_handler[sig] = signal.signal(sig, self.handler)

    def handler(self, sig, frame):
        self.signal_received[sig] = frame
        print('Delaying received signal', sig)

    def __exit__(self, type, value, traceback):
        for sig in self.SIGNALS:
            signal.signal(sig, self.old_handler[sig])
        for sig, frame in self.signal_received.items():
            old_handler = self.old_handler[sig]
            print('Resuming received signal', sig)
            if callable(old_handler):
                old_handler(sig, frame)
            elif old_handler == signal.SIG_DFL:
                sys.exit(0)
        self.signal_received = {}
        self.old_handler = {}
