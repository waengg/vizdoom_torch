import os

DEVICE_NUMBER = 0

ROOT_DIR = os.environ['VZD_TORCH_DIR']

DEFAULT_PARAMS = {
    'training_steps': 1000000,
    'doom_config': f'{ROOT_DIR}/scenarios/configs/basic.cfg',
    'mem_size': 100000,
    'dry_size': 2000,
    'input_shape': (4, 64, 64)
}
