"""MLP 64-16-4, the narrow shape, with hidden layers of 64, 16 and 4 units."""

from functools import partial
from models import shape

# Hidden layer widths
HIDDEN = (64, 16, 4)

# Settings and fitting shared by every shape
GRID = shape.grid()
build = partial(shape.build, hidden=HIDDEN)
fit = shape.fit
finalise = shape.finalise
score = shape.score
importance = shape.importance
regularisation = shape.regularisation
