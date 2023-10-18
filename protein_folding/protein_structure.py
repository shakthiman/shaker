import collections

# Structure Data Structures
Atom = collections.namedtuple('Atom', ['name', 'coord'])
Residue = collections.namedtuple('Residue', ['name', 'atoms'])
Structure = collections.namedtuple('Structure', ['name', 'residues'])


# Converts a Bio.PDB.Structure.Structure to Structure
def StructureFromPDBStructure(structure_name, pdb_structure):
  residues = []
  for r in pdb_structure.get_residues():
    atoms = []
    for a in r.get_atoms():
      atoms.append(Atom(a.name, a.get_coord().tolist()))
    residues.append(Residue(r.get_resname(), atoms))
  return Structure(structure_name, residues)

# Avro Schema Definitions
def AtomAvroSchema():
    return {
        "namespace": "ml_fun.protein_structure",
        "type": "record",
        "name": "Atom",
        "fields": [
            {"name": "name", "type": "string"},
            {"name": "coord",  "type": {"type": "array", "items": "float"}},
        ] 
    }

def ResidueAvroSchema():
    return {
        "namespace": "ml_fun.protein_structure",
        "type": "record",
        "name": "Residue",
        "fields": [
            {"name": "name", "type": "string"},
            {"name": "atoms",  "type": {"type": "array", "items": "Atom"}},
        ]
    }

def StructureAvroSchema():
    return {
        "namespace": "ml_fun.protein_structure",
        "type": "record",
        "name": "Structure",
        "fields": [
            {"name": "name", "type": "string"},
            {"name": "residues",  "type": {"type": "array", "items": "Residue"}},
        ]
    }
    
def AllAvroSchemas():
    return [AtomAvroSchema(), ResidueAvroSchema(), StructureAvroSchema()]
                   
# Converts the Structures to Avro JSon Objects
def ConvertAtomToAvro(atom):
  return {"name": atom.name, "coord": atom.coord}

def ConvertResidueToAvro(residue):
  return {"name": residue.name,
          "atoms": [ConvertAtomToAvro(a) for a in residue.atoms]}
                   
def ConvertStructureToAvro(structure):
  return {"name": structure.name,
          "residues": [ConvertResidueToAvro(r) for r in structure.residues]}