"""A module containing helper functions accessed by multiple classes.

Attributes:
    SELECTORS: Pairs of VMD selection keywords and the corresponding
        attribute and type to check in a system.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from ..constants import Subsystem

if TYPE_CHECKING:
    from pydft_qmmm import System

SELECTORS = {
    "element": ("elements", str),
    "atom": ("atoms", int),
    "index": ("atoms", int),
    "name": ("names", str),
    "label": ("labels", str),
    "residue": ("residues", int),
    "resid": ("residues", int),
    "resname": ("residue_names", str),
    "subsystem": ("subsystems", Subsystem),
}

SELECT_KEYWORDS = {
    "and": (["and", "or"], lambda sel, pred: sel & pred),
    "or": (["or"], lambda sel, pred: sel | pred),
}

VARIABLES = {
    "x": ("positions", [0]),
    "y": ("positions", [1]),
    "z": ("positions", [2]),
    "vx": ("velocities", [0]),
    "vy": ("velocities", [1]),
    "vz": ("velocities", [2]),
    "fx": ("forces", [0]),
    "fy": ("forces", [1]),
    "fz": ("forces", [2]),
    "mass": ("masses", []),
    "charge": ("charges", []),
}

OPERATORS = {
    "^": (["+", "-", "*", "/", "^", "=", ">=", "<=", ">", "<"], "__pow__"),
    "*": (["+", "-", "*", "/", "=", ">=", "<=", ">", "<"], "__mul__"),
    "/": (["+", "-", "*", "/", "=", ">=", "<=", ">", "<"], "__truediv__"),
    "+": (["+", "-", "=", ">=", "<=", ">", "<"], "__add__"),
    "-": (["+", "-", "=", ">=", "<=", ">", "<"], "__sub__"),
    "=": ([], "__eq__"),
    ">=": ([], "__ge__"),
    "<=": ([], "__le__"),
    ">": ([], "__gt__"),
    "<": ([], "__lt__"),
}

FUNCTIONS = {
    "sqrt": "sqrt",
    "sqr": "sqrt",
    "abs": "abs",
}

MATH_KEYWORDS = VARIABLES.keys() | OPERATORS.keys() | FUNCTIONS.keys()

KEYWORDS = MATH_KEYWORDS | SELECT_KEYWORDS.keys() | {"(", ")", "not", "within", "of", "same", "as"}


def isvalue(text: str) -> bool:
    if text.count(".") > 1:
        return False
    numbers = [a.isnumeric() for a in text.split(".") if a]
    if len(numbers) == 0:
        return False
    return all(numbers)


def decompose(text: str) -> list[str]:
    """Decompose an atom selection query into meaningful components.

    Args:
        text: The atom selection query.

    Returns:
        The atom selection query broken into meaningful parts,
        demarcated by keywords.
    """
    criteria = (r"(not| or | and |\(|\)|within| of |same| as "
                + r"".join([rf"|\{x}" for x in OPERATORS.keys()])
                + r")")
    line = [a.strip() for a in re.split(criteria, text)]
    while "" in line:
        line.remove("")
    return line


def line_slice(line: list[str], start: int, low_priority: list[str] = []) -> slice:
    """Find the slice of a query within parentheses.

    Args:
        line: The atom selection query, broken into meaningful
            components.
        start: The index of the line where the statement within
            parentheses begins.

    Returns:
        The slice whose start and stop corresponds to the phrase
        contained by parentheses.
    """
    count_dict = {"(": 1, ")": -1}
    flag = True
    count = count_dict.get(line[start], 0)
    index = start + 1
    while flag and index < len(line):
        count += count_dict.get(line[index], 0)
        if (count == 0 and line[index] in low_priority
            # This allows precedence of unary operators.
            and index > start + 1):
            flag = False
        else:
            index += 1
    if count > 0:
        raise ValueError("Unclosed parenthesis in atom selection query")
    return slice(start + 1, index)


def evaluate(text: str, system: System) -> frozenset[int]:
    """Evaluate a part of an atom selection query.

    Args:
        text: A single contained statement from an atom selection query.
        system: The system whose atoms will be selected by evaluating
            a single query statement.

    Returns:
        The set of atom indices selected by the query statement.
    """
    line = text.split(" ")
    category = SELECTORS[line[0].lower()]
    if " ".join(line).lower().startswith("atom name"):
        category = SELECTORS["name"]
        del line[1]
    elif " ".join(line).lower().startswith("residue name"):
        category = SELECTORS["resname"]
        del line[1]
    ret: frozenset[int] = frozenset({})
    if category[0] == "atoms":
        for string in line[1:]:
            value = category[1](string)
            ret = ret | frozenset({value})
    else:
        population = getattr(system, category[0])
        for string in line[1:]:
            value = category[1](string)
            indices = {i for i, x in enumerate(population) if x == value}
            ret = ret | frozenset(indices)
    return ret


def evaluate_math(line: list[str], system: System) -> NDArray[np.float64]:
    value = np.zeros((len(system),))
    count = 0
    entry = line[count]
    if entry.lower() in VARIABLES:
        var = VARIABLES[entry.lower()]
        value += getattr(system, var[0])[:, *var[1]]
    elif isvalue(entry):
        value += float(entry)
    elif entry.split(" ")[0].lower() in SELECTORS:
        entry = entry.split(" ")
        category = SELECTORS[entry[0].lower()]
        if category[0] == "atoms":
            value += np.array([i for i in range(len(system))])
        elif category[0] == "residues":
            value += system.residues
        else:
            raise TypeError
    elif entry in KEYWORDS:
        if entry in FUNCTIONS:
            indices = line_slice(line, count, OPERATORS.keys())
            predicate = evaluate_math(line[indices], system)
            value += getattr(np, FUNCTIONS[entry])(predicate)
        elif entry in ["+", "-"]:
            indices = line_slice(line, count, ["+", "-", "*", "/", "^"])
            predicate = evaluate_math(line[indices], system)
            value += predicate if entry == "+" else -predicate
        elif entry == "(":
            indices = line_slice(line, count, [")"])
            value += evaluate_math(line[indices], system)
        else:
            raise ValueError(
                ("Two incompatable math operators have been placed "
                 "next to each other in a query."),
            )
        count = indices.stop - 1
    else:
        raise ValueError(f"{entry =}")
    while count < len(line) - 1:
        count += 1
        entry = line[count]
        if entry in OPERATORS:
            operator = OPERATORS[entry]
            indices = line_slice(line, count, operator[0])
            predicate = evaluate_math(line[indices], system)
            value = getattr(value, operator[1])(predicate)
            count = indices.stop - 1
        elif entry not in KEYWORDS:
            raise ValueError(f"{entry =}")
    return value


def interpret(line: list[str], system: System) -> frozenset[int]:
    """Interpret a line of atom selection query language.

    Args:
        line: The atom selection query, broken into meaningful
            components.
        system: The system whose atoms will be selected by interpreting
            the selection query.

    Returns:
        The set of atom indices selected by the query.

    .. note:: Based on the VMD atom selection rules.
    """
    selection: frozenset[int] = frozenset({})
    count = 0
    entry = line[count]
    if entry.split(" ")[0].lower() in SELECTORS:
        indices = line_slice(line, count - 1, ["and", "or"])
        if any([x in MATH_KEYWORDS
                for x in line[indices]]):
            indices = slice(0, indices.stop)
            selection = selection | set(np.where(evaluate_math(line[indices], system))[0])
        else:
            selection = selection | evaluate(entry, system)
        count = indices.stop - 1
    elif entry == "all":
        selection = selection | frozenset(range(len(system)))
    elif entry == "none":
        selection = selection | frozenset({})
    elif entry in KEYWORDS or isvalue(entry):
        if entry == "(":
            indices = line_slice(line, count, [")"])
            if all([isvalue(x) or x in MATH_KEYWORDS
                    for x in line[indices]]):
                indices = line_slice(line, count, ["and", "or"])
                indices = slice(0, indices.stop)
                selection = selection | set(np.where(evaluate_math(line[indices], system))[0])
            else:
                selection = selection | interpret(line[indices], system)
        elif entry == "not":
            indices = line_slice(line, count, ["and", "or"])
            selection = selection | (frozenset(range(len(system))) - interpret(line[indices], system))
        elif entry == "within":
            # This does not currently support PBC, as in VMD.
            indices = line_slice(line, count, ["of"])
            radius = evaluate_math(line[indices], [0])[0]
            atoms = sorted(interpret(line[indices.stop+1:], system))
            measure = np.min(
                np.linalg.norm(
                    (system.positions.base[:, np.newaxis, :]
                     - system.positions[atoms, :]),
                    axis=2,
                ),
                axis=1,
            )
            selection = selection | set(np.where(measure < radius)[0])
            indices = line_slice(line, count)
        elif entry == "same":
            attribute = line[count+1]
            atoms = sorted(interpret(line[count+3:], system))
            if attribute.lower() in SELECTORS:
                text = attribute.split(" ")
                category = SELECTORS[text[0].lower()]
                if " ".join(text).lower().startswith("atom name"):
                    category = SELECTORS["name"]
                elif " ".join(text).lower().startswith("residue name"):
                    category = SELECTORS["resname"]
                if category[0] != "atoms":
                    population = getattr(system, category[0])
                    atoms = {i for i, x in enumerate(population) if x in population[atoms]}
                else:
                    atoms = set(atoms)
            elif attribute.lower() in VARIABLES:
                var = VARIABLES[attribute.lower()]
                value = getattr(system, var[0])[:, *var[1]]
                atoms = {i for i, x in enumerate(value) if x in value[atoms]}
            else:
                raise ValueError(f"Unrecognized attribute '{attribute}'")
            selection = selection | atoms
            indices = line_slice(line, count)
        elif entry in MATH_KEYWORDS or isvalue(entry):
            indices = line_slice(line, count, ["and", "or"])
            indices = slice(0, indices.stop)
            selection = selection | set(np.where(evaluate_math(line[indices], system))[0])
        count = indices.stop - 1
    else:
        raise ValueError(f"{entry =}")
    while count < len(line) - 1:
        count += 1
        entry = line[count]
        if entry in SELECT_KEYWORDS:
            keyword = SELECT_KEYWORDS[entry]
            indices = line_slice(line, count, keyword[0])
            predicate = interpret(line[indices], system)
            selection = keyword[1](selection, predicate)
            count = indices.stop - 1
        elif entry not in KEYWORDS:
            raise ValueError(f"{entry =}")
    return selection
