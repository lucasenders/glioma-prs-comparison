"""MLP 256-64, the shallow shape, with hidden layers of 256 and 64 units."""

from functools import partial
from models import shape

# Hidden layer widths
HIDDEN = (256, 64)

# Settings and fitting shared by every shape
GRID = shape.grid()
build = partial(shape.build, hidden=HIDDEN)
fit = shape.fit
finalise = shape.finalise
score = shape.score
importance = shape.importance
regularisation = shape.regularisation
