import os


ROOT_DIR = os.environ['VZD_TORCH_DIR']

DEFAULT_PARAMS = {
    'episodes': 10000,
    'training_steps': 1000000,
    'batch_size': 32,
    'doom_config': f'{ROOT_DIR}/configs/e1m1.cfg',
    'mem_size': 200000,
    'dry_size': 10000,
    'input_shape': (4, 64, 64),
    'history': 4,
    'frameskip': 4,
    'CNN': {
        'input_shape': (4, 64, 64)
    }
}
