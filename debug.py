from config_reader import ConfigReader
from common import Singleton


class Debug(metaclass=Singleton):
    def __init__(self):
        self.debug_config = ConfigReader('debug')

    def log(self, message):
        if self.debug_config.config['log'] == 'LOG_ALL':
            print(message)
