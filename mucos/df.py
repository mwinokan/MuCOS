import molparse as mp
import pandas as pd
import mrich
from rdkit import Chem
from rdkit.Chem import AllChem


def combined_dataframe(inspiration_mols, derivative_mol, remove_hs=True):

    data = []
    mols = inspiration_mols + [derivative_mol]
    for j, mol in enumerate(mols):

        if remove_hs:
            mol = Chem.RemoveHs(mol)

        group = mp.rdkit.mol_to_AtomGroup(mol)

        AllChem.ComputeGasteigerCharges(mol)

        for i, (mp_atom, rd_atom) in enumerate(zip(group.atoms, mol.GetAtoms())):

            data.append(
                dict(
                    mol_index=j,
                    atom_index=i,
                    derivative=j == len(mols) - 1,
                    species=mp_atom.species,
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

    df = pd.DataFrame(data)

    return df.set_index(["mol_index", "atom_index"])
