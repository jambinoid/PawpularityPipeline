import tensorflow as tf


_BACKBONES = ['B0', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']


def EfficientNet(
    backbone: str,
    input_height: int,
    input_width: int,
    weights: str = 'imagenet',
    dropout_rate: float = 0.2,
    pooling: str = 'avg',
    regression: bool = True,
    n_classes: int = 1
):
    if backbone not in _BACKBONES:
        raise ValueError(
            f'There is no such backbone as {backbone}. Must be one of B0-B7')

    name = f'EfficientNet{backbone}'
    model = eval(f'tf.keras.applications.efficientnet.{name}')(
        include_top=False,
        weights=weights,
        input_shape=(input_height, input_width, 3),
    )
    
    x = model.output
    
    if pooling == 'avg':
        x = tf.keras.layers.GlobalAveragePooling2D() (x)
    elif pooling == 'max':
        x = tf.keras.layers.GlobalMaxPooling2D() (x)
    
    x = tf.keras.layers.Dropout(dropout_rate, name='top_dropout') (x)
    
    if regression:
        x = tf.keras.layers.Dense(1, activation='relu', name='predictions') (x)
    else:
        if n_classes == 1:
            x = tf.keras.layers.Dense(1, activation='sigmoid', name='predictions') (x)
        else:
            x = tf.keras.layers.Dense(n_classes, activation='softmax', name='predictions') (x)
            
    return tf.keras.Model(inputs=model.inputs, outputs=[x], name=name)