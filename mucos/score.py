import mrich
from rdkit import Chem

from .df import combined_dataframe
from .shape import known_inspiration_shape


def known_inspiration_score(
    inspirations: list[Chem.Mol],
    derivative: Chem.Mol,
    debug: bool = True,
) -> float:

    if debug:
        mrich.debug("known_inspiration_score()")

    df = combined_dataframe(inspirations, derivative)
    shape_score = known_inspiration_shape(df)

    if debug:
        mrich.debug("shape_score", shape_score)

    return df
