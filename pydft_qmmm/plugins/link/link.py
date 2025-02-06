"""Plugin for computing region I-region II bonded interactions with the link-atom method
"""
from __future__ import annotations

from typing import Callable
from typing import TYPE_CHECKING

import numpy as np

from pydft_qmmm.calculators import InterfaceCalculator
from pydft_qmmm.interfaces import QMInterface
from pydft_qmmm.interfaces import MMInterface
from pydft_qmmm.plugins.plugin import CompositeCalculatorPlugin

if TYPE_CHECKING:
    from pydft_qmmm.calculators import CompositeCalculator
    from pydft_qmmm.common import Results
    import mypy_extensions
    CalculateMethod = Callable[
        [
            mypy_extensions.DefaultArg(
                bool | None,
                "return_forces", # noqa: F821
            ),
            mypy_extensions.DefaultArg(
                bool | None,
                "return_components", # noqa: F821
            ),
        ],
        Results,
    ]

class LINK(CompositeCalculatorPlugin):
    def __init__(
            self,
            boundary_atoms = list[tuple[int,list[int]]]
    ) -> None:
        self._boundary_atoms = boundary_atoms

    def modify(
        self,
        calculator: CompositeCalculator,
    ) -> None:
        """Modify the functionality of a calculator.

        Args:
            calculator: The calculator whose functionality will
                modified by the plugin.
        """
        self._modifieds.append(type(calculator).__name__)
        self.system = calculator.system
        # Grab the QM and MM calculators so we can do them separately
        for calc in calculator.calculators:
            if isinstance(calc, InterfaceCalculator):
                if isinstance(calc.interface, QMInterface):
                    self.qm_interface = calc.interface
                elif isinstance(calc.interface, MMInterface):
                    self.mm_interface = calc.interface
