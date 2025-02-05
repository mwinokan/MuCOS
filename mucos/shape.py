import mrich
import numpy as np
from pandas import DataFrame

from .gaussian import spherical_gaussian_overlap, spherical_gaussian_union_volume
from .similarity import jaccard


def known_inspiration_shape(
    df: DataFrame,
    debug: bool = True,
) -> float:

    if debug:
        mrich.debug("known_inspiration_shape()")

    # get dataframe subsets
    df = df[df["category"] == "atom"]
    inspiration_df = df[~df["derivative"]]
    derivative_df = df[df["derivative"]]

    # get atom counts
    inspiration_volume = calculate_molecular_volume(inspiration_df)
    derivative_volume = calculate_molecular_volume(derivative_df)

    overlaps = calculate_overlaps(derivative_df, inspiration_df)
    derivative_volume_overlap = sum(overlaps)

    if debug:
        mrich.debug("inspiration_volume", inspiration_volume)
        mrich.debug("derivative_volume", derivative_volume)
        mrich.debug("derivative_volume_overlap", derivative_volume_overlap)

    # Use the Jaccard index (or Tversky w/ alpha=beta=1)
    return jaccard(
        inspiration_volume,
        derivative_volume,
        derivative_volume_overlap,
    )


def calculate_overlaps(df1, df2) -> list[float]:

    best_overlaps = []

    for i, row1 in df1.iterrows():

        pos1 = np.array([row1.x, row1.y, row1.z])
        rad1 = row1.vdw_radius

        overlaps = []

        for j, row2 in df2.iterrows():

            pos2 = np.array([row2.x, row2.y, row2.z])
            rad2 = row2.vdw_radius

            dist = np.linalg.norm(pos2 - pos1)

            overlap = spherical_gaussian_overlap(rad1, rad2, dist)

            overlaps.append(overlap)

        best_overlaps.append(max(overlaps))

    return best_overlaps


def calculate_molecular_volume(df):

    # assume that within each molecule there are no third order or higher intersections

    df = df.reset_index()

    mol_indices = set(df["mol_index"].to_list())

    volumes = []

    for mol_index in mol_indices:

        subset = df[df["mol_index"] == mol_index]

        centres = []
        radii = []

        for i, row in subset.iterrows():
            centres.append(np.array([row.x, row.y, row.z]))
            radii.append(row.vdw_radius)

        volumes.append(spherical_gaussian_union_volume(centres, radii))

    # n_molecules = len(df.index.levels[0])

    # display(mol_ids)

    # print(n_molecules)
    # print(df.index.levels[0])

    # molecular_volumes
