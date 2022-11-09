import argparse


def add_arguments(parser):
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--model_type', type=str, default='')


def run(args, state):
    while True:
        pass


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    run(args, None)


if __name__ == '__main__':
    main()
