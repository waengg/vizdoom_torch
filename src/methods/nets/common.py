import os

DEVICE_NUMBER = 0

ROOT_DIR = os.path.abspath

DEFAULT_PARAMS = {
    'training_steps': 1000000,
    'doom_config': f'{ROOT_DIR}/scenarios/configs/basic.cfg'
}
