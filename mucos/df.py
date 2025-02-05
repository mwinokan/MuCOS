import molparse as mp
import pandas as pd
import mrich
from rdkit import Chem
from rdkit.Chem import AllChem


def combined_dataframe(
    inspiration_mols: list[Chem.Mol],
    derivative_mol: Chem.Mol,
    # remove_hs: bool = True,
) -> pd.DataFrame:

    """Creates a DataFrame combining all atoms and pharmacophoric features from the provided molecules"""

    data = []
    mols = inspiration_mols + [derivative_mol]

    for j, mol in enumerate(mols):

        # if remove_hs:
        #     mol = Chem.RemoveHs(mol)

        group = mp.rdkit.mol_to_AtomGroup(mol)

        AllChem.ComputeGasteigerCharges(mol)

        is_derivative = j == len(mols) - 1

        for i, (mp_atom, rd_atom) in enumerate(zip(group.atoms, mol.GetAtoms())):

            species = mp_atom.species

            if species == "H":
                continue

            data.append(
                dict(
                    mol_index=j,
                    atom_index=i,
                    derivative=is_derivative,
                    category="atom",
                    species=species,
                    x=mp_atom.x,
                    y=mp_atom.y,
                    z=mp_atom.z,
                    vdw_radius=mp_atom.vdw_radius,
                    covalent_radius=mp_atom.covalent_radius,
                    degree=rd_atom.GetDegree(),
                    hybridization=rd_atom.GetHybridization(),
                    is_aromatic=rd_atom.GetIsAromatic(),
                    formal_charge=rd_atom.GetFormalCharge(),
                    gasteiger_charge=rd_atom.GetProp("_GasteigerCharge"),
                )
            )

        for feature in mp.rdkit.features_from_mol(mol, group=group, protonate=True):

            if len(feature.atoms) > 1:
                entry_type = "lumped pharmacophore"
                atom_index = tuple(
                    set(sorted([atom.index - 1 for atom in feature.atoms]))
                )
            else:
                entry_type = "pharmacophore"
                atom_index = feature.atoms[0].index - 1

            data.append(
                dict(
                    mol_index=j,
                    atom_index=atom_index,
                    derivative=is_derivative,
                    species=feature.family,
                    category=entry_type,
                    x=feature.x,
                    y=feature.y,
                    z=feature.z,
                )
            )

    df = pd.DataFrame(data)

    # return df.set_index(["mol_index"])
    return df.set_index(["mol_index", "atom_index"])
