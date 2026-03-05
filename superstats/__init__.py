def setup():
    # placeholder for now
    pass

# call and clean up namespace
setup()
del setup

from . import (
    prior,
    simulation,
    study,
    transition,
)
