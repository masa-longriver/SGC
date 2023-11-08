config = {
    'seed': 25,
    'train': {
        'epochs': 200,
        'patience': 10
    },
    'model': {
        'K': 2
    },
    'optim': {
        'lr': 0.01,
        'beta1': 0.9,
        'beta2': 0.999,
        'weight_decay': 5e-4,
        'eps': 1e-8
    },
}

def load_config():

    return config