from sandbox.rocky.tf.algos.trpo import TRPO

class MyAlgo(TRPO):
    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        super().__init__(optimizer, optimizer_args, **kwargs)

