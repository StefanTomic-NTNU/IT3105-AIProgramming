import yaml

from sarsa import run_algo


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def main():
    config = load_config('config.yaml')
    run_algo(config)


if __name__ == '__main__':
    main()
