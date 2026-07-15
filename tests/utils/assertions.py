import keras


def assert_layers_equal(layer1: keras.Layer, layer2: keras.Layer):
    msg = f"Layers {layer1.name} and {layer2.name} have different types."
    assert type(layer1) is type(layer2), msg

    msg = (
        f"Layers {layer1.name} and {layer2.name} have a different number of variables "
        f"({len(layer1.variables)}, {len(layer2.variables)})."
    )
    assert len(layer1.variables) == len(layer2.variables), msg

    msg = f"Layers {layer1.name} and {layer2.name} have different build status: {layer1.built} != {layer2.built}"
    assert layer1.built == layer2.built, msg

    for v1, v2 in zip(layer1.variables, layer2.variables):
        if v1.name == "seed_generator_state":
            continue

        x1 = keras.ops.convert_to_numpy(v1)
        x2 = keras.ops.convert_to_numpy(v2)
        msg = f"Variable '{v1.name}' for Layer '{layer1.name}' is not equal: {x1} != {x2}"
        assert keras.ops.all(keras.ops.isclose(x1, x2)), msg
