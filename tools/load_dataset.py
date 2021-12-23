import argparse

from pawpularitypipeline.datasets import PawpularityDatasetFactory


def main():
    dataset_factory = PawpularityDatasetFactory()
    print(f'Start sync up')
    dataset_factory.sync()
    print(f'{dataset_factory._DATASET_DIR} is synced up with kaggle')


if __name__ == '__main__':
    main()