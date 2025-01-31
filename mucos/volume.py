import molparse as mp
import mrich
import numpy as np
import pandas as pd


def display_overlaps(inspiration_mols, derivative_mol):

    mp.rdkit.draw_mols(inspiration_mols + [derivative_mol])

    from .df import combined_dataframe

    df = combined_dataframe(inspiration_mols, derivative_mol)

    derivative_volume_df = annotate_overlaps(df)
    inspiration_volume_df = annotate_overlaps(df, invert=True)

    for i, mol in enumerate(inspiration_mols):

        mol._Name = f"inspiration {i+1}"

        mrich.h3(mol._Name)

        subdf = inspiration_volume_df.xs(i)

        colors = [
            (j, (0, 0.5, 0)) for j in subdf[subdf["intersection_threshold"]].index
        ]
        colors += [
            (j, (0.5, 0, 0)) for j in subdf[~subdf["intersection_threshold"]].index
        ]

        display(mp.rdkit.draw_highlighted_mol(mol, colors, flat=True))

    colors = [
        (j, (0, 0.5, 0))
        for i, j in derivative_volume_df[
            derivative_volume_df["intersection_threshold"]
        ].index
    ]
    colors += [
        (j, (0.5, 0, 0))
        for i, j in derivative_volume_df[
            ~derivative_volume_df["intersection_threshold"]
        ].index
    ]

    derivative_mol._Name = "derivative"
    mrich.h3(derivative_mol._Name)
    display(mp.rdkit.draw_highlighted_mol(derivative_mol, colors, flat=True))


def annotate_overlaps(df, invert: bool = False) -> "pd.DataFrame":

    results = []

    if invert:
        derivative_df = df[~df["derivative"]]
        inspiration_df = df[df["derivative"]]
    else:
        derivative_df = df[df["derivative"]]
        inspiration_df = df[~df["derivative"]]

    for i, row in derivative_df.iterrows():

        # mrich.print(i, row.species, row.vdw_radius)

        df = inspiration_df.copy()

        df["distance"] = df.apply(compute_distance, axis=1, reference=row)
        df["overlap"] = df.apply(compute_overlap, axis=1, reference=row)
        df["within"] = df.apply(compute_within, axis=1, reference=row)
        df["vdw_intersection"] = df.apply(
            compute_vdw_intersection, axis=1, reference=row
        )
        df["intersection_fraction"] = df.apply(
            compute_intersection_fraction, axis=1, reference=row
        )

        results.append(
            dict(
                mol_index=i[0],
                atom_index=i[1],
                overlaps=len(df[df["overlap"]]),
                withins=len(df[df["within"]]),
                max_intersection_fraction=max(df["intersection_fraction"].values),
            )
        )

    all_df = pd.DataFrame(results)

    all_df["intersection_threshold"] = all_df["max_intersection_fraction"] >= 0.5

    if invert:
        mrich.var(
            "#inspiration atoms within derivative",
            len(all_df[all_df["intersection_threshold"]]),
        )
        mrich.var(
            "#inspiration atoms not within derivative",
            len(all_df[~all_df["intersection_threshold"]]),
        )
    else:
        mrich.var(
            "#derivative atoms within inspirations",
            len(all_df[all_df["intersection_threshold"]]),
        )
        mrich.var(
            "#derivative atoms not within inspirations",
            len(all_df[~all_df["intersection_threshold"]]),
        )

    all_df = all_df.set_index(["mol_index", "atom_index"])

    return all_df


def compute_distance(row, reference):
    reference = np.array([reference["x"], reference["y"], reference["z"]])
    point = np.array([row["x"], row["y"], row["z"]])
    return np.linalg.norm(point - reference)


def compute_overlap(row, reference):
    return (reference["covalent_radius"] + row["covalent_radius"]) >= row["distance"]


def compute_within(row, reference, tolerance=0.5):
    return tolerance + row["vdw_radius"] >= reference["vdw_radius"] + row["distance"]


def compute_vdw_intersection(row, reference):
    volume = (
        np.pi
        * (
            np.power(row["distance"] + row["vdw_radius"] + reference["vdw_radius"], 2)
            * np.power(row["vdw_radius"] - reference["vdw_radius"], 2)
        )
        / (12 * row["distance"])
    )


def compute_intersection_fraction(row, reference, plot: bool = False):

    r = reference["vdw_radius"]
    R = row["vdw_radius"]
    d = row["distance"]

    if R + r < d:
        return 0

    d2 = np.power(d, 2)
    r2 = np.power(r, 2)
    R2 = np.power(R, 2)
    dr = d * r
    dR = d * R
    rR = r * R

    intersection = (
        np.pi
        * np.power(r + R - d, 2)
        * (d2 + 2 * dr - 3 * r2 + 2 * dR + 6 * rR - 3 * R2)
        / (12 * d)
    )
    volume = 4 / 3 * np.pi * np.power(R, 3)
    fraction = intersection / volume

    if plot:

        mrich.var("R", f"{R:.2f}")
        mrich.var("r", f"{r:.2f}")
        mrich.var("d", f"{d:.2f}")
        mrich.var("intersection", f"{intersection:.2f}")
        mrich.var("volume", f"{volume:.2f}")
        mrich.var("fraction", f"{fraction:.2f}")

        import plotly.graph_objects as go

        fig = go.Figure()

        fig.add_shape(
            type="circle", x0=-R, y0=-R, x1=R, y1=R, line_color="LightSeaGreen"
        )
        fig.add_shape(type="circle", x0=d - r, y0=-r, x1=d + r, y1=r, line_color="Red")

        fig.show()

    return fraction
