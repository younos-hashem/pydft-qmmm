"""Plugin for computing region I-region II bonded interactions with the link-atom method
"""
from __future__ import annotations

from typing import Callable
from typing import TYPE_CHECKING

import numpy as np

class LINK(CalculatorPlugin):
