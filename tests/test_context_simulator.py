import numpy as np

from superstats.simulation import ContextSimulator


def test_context_simulator_calls_batched_simulator_directly():
    calls = []

    def simulator(*, batch_size, num_steps):
        calls.append((batch_size, num_steps))
        return {"context": np.zeros((batch_size, num_steps))}

    result = ContextSimulator(simulator, is_batched=True).sample(batch_size=3, num_steps=4)

    assert calls == [(3, 4)]
    assert result["context"].shape == (3, 4)


def test_context_simulator_stacks_non_batched_simulations():
    calls = []

    def simulator(*, num_steps):
        calls.append(num_steps)
        return {"context": np.arange(num_steps)}

    result = ContextSimulator(simulator, is_batched=False).sample(batch_size=3, num_steps=4)

    assert calls == [4, 4, 4]
    np.testing.assert_array_equal(
        result["context"],
        np.tile(np.arange(4), (3, 1)),
    )
