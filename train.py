import argparse
from copy import copy
from functools import partial
import os
import random
import time
from typing import Dict

import numpy as np
import mlflow
import tensorflow as tf
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import yaml

from pawpularitypipeline.datasets import PawpularityDatasetFactory
from pawpularitypipeline.models.efficientnets import EfficientNet
from pawpularitypipeline.preprocessing import resize
from pawpularitypipeline.preprocessing.augmentations_combined import augment_v1


def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main(config: Dict):
    # Set seed
    set_seed(config.get('seed', 21))

    config_to_save = copy(config)

    # Get model
    model_config = config['model']
    model_name = model_config.pop('name')

    optimizer_config = config['optimizer']
    optimizer_name = optimizer_config.pop('name')

    scheduler_config = config['scheduler']
    scheduler_name = scheduler_config.pop('name')

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = eval(model_name)(**model_config)
        if scheduler_name == 'const':
            scheduler = scheduler_config.get('lr', 1e-3)
        else:
            scheduler = eval(scheduler_name)(**scheduler_config) 
        optimizer = eval(optimizer_name)(
            lr=scheduler, **optimizer_config)
        loss = eval(config['loss'])()
        metric = eval(config['metric'])()

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[metric],
        )

    print(model.summary())

    # Get datasets
    dataset_factory = PawpularityDatasetFactory()
    preproc_func = partial(
        eval(config.get('preprocessor')),
        height=model_config['input_height'],
        width=model_config['input_width']
    )

    train_dataset_config = config['train_dataset']
    train_dataset, val_dataset = dataset_factory.get_train_dataset(
        aug_func=eval(config.get('augmentations')),
        preproc_func=preproc_func,
        **train_dataset_config
    )
    test_dataset_config = config['test_dataset']
    test_dataset = dataset_factory.get_test_dataset(
        preproc_func=preproc_func,
        **test_dataset_config
    )

    # Init MLFlow
    mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
    mlflow.set_experiment('Pawpularity')
    run_name = model_name + '-' + time.strftime('%Y%_%m_%d_%H')
    mlflow.start_run(run_name=run_name) 

    mlflow.log_params(config_to_save)

    mlflow.tensorflow.autolog()

    # Set callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]

    # Train model
    train_config = config['train']
    model.fit(
        x=train_dataset,
        validation_data=val_dataset,
        callbacks=callbacks,
        **train_config
    )

    # Evaluate on test data
    test_metrics = model.evaluate(
        x=test_dataset,
        use_multiprocessing=train_config['use_multiprocessing']
    )
    print(test_metrics)
    mlflow.log_metric('test_loss', test_metrics[0])
    mlflow.log_metric(f'test_{metric.name}', test_metrics[1])

    mlflow.end_run()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser('Train model using config')
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    # Load .yaml config
    with open(args.config) as f:
        config = yaml.safe_load(f)
        # config = dacite.from_dict(data_class=TrainConfig, data=config)

    main(config)