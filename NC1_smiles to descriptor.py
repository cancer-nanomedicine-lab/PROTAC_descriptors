# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:02:52 2024

@author: PanagioM
"""

from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Read the input CSV file that contains drug names and SMILES
input_filename = 'NC1_INPUT_drug list.csv'
drug_list = pd.read_csv(input_filename)

# Initialize Mordred calculator
calc_2d = Calculator(descriptors, ignore_3D=True)
calc_3d = Calculator(descriptors, ignore_3D=False)

# Generate RDKit molecule objects from SMILES and generate 3D conformers
comps = []
none_summary = []

for i, smi in tqdm(enumerate(drug_list['Smiles']), desc="Generating 3D Conformers", total=len(drug_list)):
    mol_2d = None
    mol_3d = None
    if pd.notnull(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mol = Chem.AddHs(mol)  # Add hydrogens
            mol_2d = mol  # Save for 2D descriptor calculation
            try:
                AllChem.EmbedMolecule(mol)  # Generate 3D coordinates
                AllChem.UFFOptimizeMolecule(mol)  # Optimize the 3D conformer
                mol_3d = mol  # Save for 3D descriptor calculation
            except ValueError as e:
                print(f"Error generating 3D conformer for molecule {drug_list['Name'][i]}: {e}")
                none_summary.append((drug_list['Name'][i], f"3D Conformer error: {e}"))
        else:
            none_summary.append((drug_list['Name'][i], 'Invalid SMILES'))
    else:
        none_summary.append((drug_list['Name'][i], 'No SMILES'))
    
    comps.append((mol_2d, mol_3d))

# Initialize list to collect descriptor dictionaries
descript_list = []

# List of problematic descriptors
problematic_descriptors = set([
    "MAXsLi", "MAXssBe", "MAXssssBe", "MAXssBH", "MAXsssB", "MAXssssB", 
    "MAXsCH3", "MAXdCH2", "MAXssCH2", "MAXtCH", "MAXdsCH", "MAXaaCH", 
    "MAXsssCH", "MAXddC", "MAXtsC", "MAXdssC", "MAXaasC", "MAXaaaC", 
    "MAXssssC", "MAXsNH3", "MAXsNH2", "MAXssNH2", "MAXdNH", "MAXssNH", 
    "MAXaaNH", "MAXtN", "MAXsssNH", "MAXdsN", "MAXaaN", "MAXsssN", 
    "MAXddsN", "MAXaasN", "MAXssssN", "MAXsOH", "MAXdO", "MAXssO", 
    "MAXaaO", "MAXsF", "MAXsSiH3", "MAXssSiH2", "MAXsssSiH", "MAXssssSi", 
    "MAXsPH2", "MAXssPH", "MAXsssP", "MAXdsssP", "MAXsssssP", "MAXsSH", 
    "MAXdS", "MAXssS", "MAXaaS", "MAXdssS", "MAXddssS", "MAXsCl", 
    "MAXsGeH3", "MAXssGeH2", "MAXsssGeH", "MAXssssGe", "MAXsAsH2", 
    "MAXssAsH", "MAXsssAs", "MAXsssdAs", "MAXsssssAs", "MAXsSeH", 
    "MAXdSe", "MAXssSe", "MAXaaSe", "MAXdssSe", "MAXddssSe", "MAXsBr", 
    "MAXsSnH3", "MAXssSnH2", "MAXsssSnH", "MAXssssSn", "MAXsI", 
    "MAXsPbH3", "MAXssPbH2", "MAXsssPbH", "MAXssssPb", "MINsLi", 
    "MINssBe", "MINssssBe", "MINssBH", "MINsssB", "MINssssB", "MINsCH3", 
    "MINdCH2", "MINssCH2", "MINtCH", "MINdsCH", "MINaaCH", "MINsssCH", 
    "MINddC", "MINtsC", "MINdssC", "MINaasC", "MINaaaC", "MINssssC", 
    "MINsNH3", "MINsNH2", "MINssNH2", "MINdNH", "MINssNH", "MINaaNH", 
    "MINtN", "MINsssNH", "MINdsN", "MINaaN", "MINsssN", "MINddsN", 
    "MINaasN", "MINssssN", "MINsOH", "MINdO", "MINssO", "MINaaO", 
    "MINsF", "MINsSiH3", "MINssSiH2", "MINsssSiH", "MINssssSi", "MINsPH2", 
    "MINssPH", "MINsssP", "MINdsssP", "MINsssssP", "MINsSH", "MINdS", 
    "MINssS", "MINaaS", "MINdssS", "MINddssS", "MINsCl", "MINsGeH3", 
    "MINssGeH2", "MINsssGeH", "MINssssGe", "MINsAsH2", "MINssAsH", 
    "MINsssAs", "MINsssdAs", "MINsssssAs", "MINsSeH", "MINdSe", "MINssSe", 
    "MINaaSe", "MINdssSe", "MINddssSe", "MINsBr", "MINsSnH3", "MINssSnH2", 
    "MINsssSnH", "MINssssSn", "MINsI", "MINsPbH3", "MINssPbH2", "MINsssPbH", 
    "MINssssPb"
])

# Calculate descriptors for each molecule and handle exceptions
for i, (mol_2d, mol_3d) in tqdm(enumerate(comps), desc="Calculating Descriptors", total=len(comps)):
    descriptors_dict = {}
    try:
        if mol_2d is not None:
            descriptors_dict.update(calc_2d(mol_2d).asdict())
    except Exception as e:
        print(f"Error calculating 2D descriptors for molecule {drug_list['Name'][i]}: {e}")
        none_summary.append((drug_list['Name'][i], f"2D Descriptor error: {e}"))
        descriptors_dict.update({desc: None for desc in calc_2d.descriptors})
    
    try:
        if mol_3d is not None:
            mol_3d_desc = calc_3d(mol_3d).asdict()
            for desc in problematic_descriptors:
                if desc in mol_3d_desc and isinstance(mol_3d_desc[desc], Exception):
                    raise mol_3d_desc[desc]
            descriptors_dict.update(mol_3d_desc)
    except Exception as e:
        print(f"Error calculating 3D descriptors for molecule {drug_list['Name'][i]}: {e}")
        none_summary.append((drug_list['Name'][i], f"3D Descriptor error: {e}"))
        descriptors_dict.update({desc: None for desc in calc_3d.descriptors})
    
    descript_list.append(descriptors_dict)

# Convert list of descriptor dictionaries to DataFrame
descript = pd.DataFrame(descript_list)

# Reset index of descript to align with drug_list
descript.reset_index(drop=True, inplace=True)

# Merge the original drug list with the descriptors DataFrame
df = pd.concat([drug_list, descript], axis=1)

# Add timestamp to the filename
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_filename = f'descriptors_reg_{timestamp}.csv'

# Save the final DataFrame to a CSV file
df.to_csv(output_filename, index=False)

# Print a summary of where None values were generated
print("\nSummary of None values generated:")
for name, reason in none_summary:
    print(f"{name}: {reason}")

# Check for None values in the DataFrame
none_values = df.isnull().sum().sum()
print(f"Total None values in the DataFrame: {none_values}")
