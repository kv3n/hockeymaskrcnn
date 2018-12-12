class ConfigReader:
    def __init__(self, filename):
        self.config = dict()

        with open('cfg/' + filename + '.cfg', 'r') as fp:
            config_lines = fp.readlines()
            for config_line in config_lines:
                config_line = config_line.rstrip('\n')
                key = config_line.split('=')[0].lstrip().rstrip()
                value = config_line.split('=')[1].lstrip().rstrip()
                self.config[key] = value

