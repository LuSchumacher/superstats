"""Defaults for data augmentation processes"""

from superstats.prior.prior import Prior

DEFAULT_P_MISSING_PRIOR = Prior("beta", a=1.5, b=15)
