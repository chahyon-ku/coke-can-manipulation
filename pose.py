import argparse


def step(args):
    pass


def init(args):
    pass


def add_arguments(parser):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    init(args)
    while True:
        step(args)
