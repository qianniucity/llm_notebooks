from pathlib import Path

import yaml


class Config:
    def __init__(self):
        self.parent_path = Path().parent.absolute()
        self.config = yaml.safe_load(open(f"{self.parent_path}/config.yaml"))
