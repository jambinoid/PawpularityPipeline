from argparse import ArgumentError
import os
import shutil
from typing import Callable, Optional, Tuple
from google.protobuf.descriptor import Error

import kaggle
import numpy as np
import pandas as pd
import tensorflow as tf


class PawpularityDatasetFactory:
    _CURRENT_DIR = os.path.realpath(__file__)
    _DATASET_DIR = os.path.join(
        '/', *_CURRENT_DIR.split('/')[:-3], '.data')

    def sync(self, test_split: int = 0.3):
        # Create directory if not exist
        if not os.path.exists(self._DATASET_DIR):
            os.mkdir(self._DATASET_DIR)
        if not os.listdir(self._DATASET_DIR):
            # Load dataset from kaggle to local directory
            author = 'schulta'
            dataset = 'petfinder-pawpularity-score-clean'
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                os.path.join(author, dataset), path=self._DATASET_DIR)
            # Extract archive with data
            shutil.unpack_archive(
                os.path.join(self._DATASET_DIR, dataset + '.zip'), self._DATASET_DIR)
            os.remove(os.path.join(self._DATASET_DIR, dataset + '.zip'))
            shutil.rmtree(os.path.join(self._DATASET_DIR, 'test'), ignore_errors=True)

            # Read data
            df_train = pd.read_csv(os.path.join(self._DATASET_DIR, 'train.csv'), index_col=0)

            # Split to train and test datasets with fixed seed
            df_test = df_train.sample(frac=test_split, random_state=21)
            df_train.drop(df_test.index, inplace=True)

            # Save .csv files with meta
            df_test.to_csv(os.path.join(self._DATASET_DIR, 'test.csv'))
            df_train.to_csv(os.path.join(self._DATASET_DIR, 'train.csv'))

    @staticmethod
    def _load_image(path: str) -> tf.Tensor:
        img = tf.io.read_file(path)
        img = tf.image.decode_image(
            img, channels=3, expand_animations=False)
        
        return img

    def _df_to_dataset(
        self,
        df: pd.DataFrame,
        preproc_func: Callable
    ) -> tf.data.Dataset:
        paths, scores = df.to_numpy().T
        paths = tf.data.Dataset.from_tensor_slices(paths.tolist())
    
        paths = (
            paths
            .map(self._load_image)
            .map(preproc_func, num_parallel_calls=tf.data.AUTOTUNE)
        )
        
        scores = tf.data.Dataset.from_tensor_slices(scores.tolist())
        
        return tf.data.Dataset.zip((paths, scores))

    def get_train_dataset(
        self,
        aug_func: Callable,
        preproc_func: Callable,
        batch_size: int,
        val_split: int = None,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        shuffle_size: int = 1000,
    ) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset]]:
        if not os.path.exists(self._DATASET_DIR):
            raise AttributeError(
                f'Directory {self._DATASET_DIR} does not exists. '
                'Create it with tools/load_dataset.py')
        if not os.listdir(self._DATASET_DIR):
            raise AttributeError(
                f'Directory {self._DATASET_DIR} is empty. '
                'Sync up using tools/load_dataset.py')

        # Read data
        df_train = pd.read_csv(os.path.join(self._DATASET_DIR, 'train.csv'), usecols=(0, 13))
        df_train.Id = df_train.Id.apply(lambda x: os.path.join(self._DATASET_DIR, 'train', x + '.jpg'))

        # Split to train and val datasets
        if val_split:
            df_val = df_train.sample(frac=val_split)
            df_train.drop(df_val.index, inplace=True)

            # Create val tf.data.Dataset
            dataset_val = self._df_to_dataset(df_val, preproc_func)
            del df_val
            
            if shuffle_val:
                dataset_val = dataset_val.shuffle(shuffle_val)

            dataset_val = (
                dataset_val
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
        else:
            dataset_val = None

        # Create train tf.data.Dataset
        dataset_train = self._df_to_dataset(df_train, preproc_func)
        del df_train

        if shuffle_train:
            dataset_train = dataset_train.shuffle(shuffle_size)

        dataset_train = (
            dataset_train
            .map(aug_func, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        return dataset_train, dataset_val

    def get_test_dataset(
        self,
        preproc_func: Callable,
        batch_size: int, 
        shuffle: bool = False,
        shuffle_size: int = 1000,
    ) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset]]:
        if not os.path.exists(self._DATASET_DIR):
            raise AttributeError(
                f'Directory {self._DATASET_DIR} does not exists. '
                'Create it with tools/load_dataset.py')
        if not os.listdir(self._DATASET_DIR):
            raise AttributeError(
                f'Directory {self._DATASET_DIR} is empty. '
                'Sync up using tools/load_dataset.py')

        # Read data
        df = pd.read_csv(os.path.join(self._DATASET_DIR, 'test.csv'), usecols=(0, 13))
        df.Id = df.Id.apply(lambda x: os.path.join(self._DATASET_DIR, 'train', x + '.jpg'))

        # Create test tf.data.Dataset
        dataset = self._df_to_dataset(df, preproc_func)
        del df

        if shuffle:
            dataset = dataset.shuffle(shuffle_size)

        dataset = (
            dataset
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        return dataset
