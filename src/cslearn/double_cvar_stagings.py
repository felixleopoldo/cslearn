"""Enumerate stagings with up to 2 context variables."""

from itertools import combinations, product
from typing import Generator, Iterable


def num_stagings(lvl: int) -> int:
    """Use formula to compute number of stagings at given level of binary CStree."""
    return lvl**3 + 1 if lvl != 2 else 8


def codim_max2_boxes(cards: Iterable, splittable_coords: Iterable[int] = None, max1cvar=False) -> Generator:
    """Enumerate ways of subdividing a given box; interpreted as stagings at a given level of a CStree.

    Args:
        cards: Cardinality of the value set for each coordinate (random variable).
        splittable_coords: Coordinate indices eligible for splitting; corresponds to
            the possible context variables for this level. ``None`` means all coordinates.
        max1cvar: If True, restrict to subdivisions with at most 1 context variable
            (codimension ≤ 1). Defaults to False (allow up to 2 context variables).

    Yields:
        list: Each yielded item is a staging — a list of sub-boxes (stages) that
        partition the full outcome space at this level.
    """

    box = [set(range(card)) for card in cards]

    codim_0_box = [box]
    yield codim_0_box

    degen = False

    dim = len(box)

    if splittable_coords is None:
        splittable_coords = list(range(dim))
    else:
        splittable_coords = list(splittable_coords)

    if not splittable_coords:
        return
    sub_split_len = len(splittable_coords) - 1
    sub_splittable_coords = reversed(tuple(combinations(splittable_coords, sub_split_len)))
    z_cd1_subdivs = zip(sub_splittable_coords, codim_1_subdivs(codim_0_box, splittable_coords))
    for poss_split_dims, cd1_subdiv in z_cd1_subdivs:
        yield cd1_subdiv
        if max1cvar:
            continue

        num_cd1_boxes = len(cd1_subdiv)
        for subset_size in range(1, num_cd1_boxes):
            subsets = combinations(range(num_cd1_boxes), subset_size)
            for subset in subsets:
                for cd12_subdiv in codim_1_subdivs(cd1_subdiv, poss_split_dims, subset):
                    yield cd12_subdiv
        if degen:
            break
        for cd2_subdiv in codim_1_subdivs(cd1_subdiv, poss_split_dims):
            yield cd2_subdiv
        if len(box) == 2:
            degen = True


def codim_1_subdivs(box: list, splittable_coords: Iterable[int], splittable_subboxes: list = None) -> Generator:
    """Enumerate codimension-1 subdivisions of a (possibly already subdivided) box.

    Args:
        box: Current subdivision — a list of sub-boxes to further subdivide.
        splittable_coords: Coordinate indices eligible for splitting.
        splittable_subboxes: Indices into ``box`` of sub-boxes that may be split.
            Defaults to all sub-boxes.

    Yields:
        list: Each yielded item is a finer subdivision of ``box``.
    """
    if splittable_subboxes is None:
        splittable_subboxes = list(range(len(box)))
    for dims_to_split in product(*(splittable_coords for _ in splittable_subboxes)):
        cd1_subdiv = []
        for subbox_idx, subbox in enumerate(box):
            if subbox_idx in splittable_subboxes:
                dim = dims_to_split[splittable_subboxes.index(subbox_idx)]
                points = box[0][dim]
                for point in points:
                    cd1_subdiv += [subbox[:dim] + [point] + subbox[dim + 1 :]]
            else:
                cd1_subdiv += [subbox]
        yield cd1_subdiv
