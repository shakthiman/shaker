import apache_beam as beam
from Bio import PDB
import collections
import numpy as np
import io
import random

import tensorflow as tf

def TrainingSummarySchema():
    return {
        "namespace": "ml_fun.training_example",
        "type": "record",
        "name": "TrainingSummary",
        "fields": [
            {"name": "residue_names", "type": {
                "type": "array",
                "items": "string"}},
            {"name": "atom_names", "type": {
                "type": "array",
                "items": "string"}},
        ]
    }

def PreProcessPDBStructure(pdb_structure):
    residue_names = []
    atom_names = []
    coords = []
    for r in pdb_structure.get_residues():
        for a in r.get_atoms():
            residue_names.append(r.get_resname())
            atom_names.append(a.get_name())
            coords.append(a.get_coord())
    residue_names = np.array(residue_names)
    atom_names = np.array(atom_names)
    normalized_coordinates = np.array(coords)
    normalized_coordinates -= np.mean(coords, 0)

    
    return {
        'name': pdb_structure.get_id(),
        'residue_names': residue_names,
        'atom_names': atom_names,
        'normalized_coordinates': normalized_coordinates,
    }

# Things to record:
#  - Chain Id [List of Chain-ids]
#  - Residue Id [Key by (Chain-idx, atom-idx)]
#  - Residue Names [Key by (Chain-idx, atom-idx)]
#  - Atom Names [Key by (Chain-idx, atom-idx)]
#  - Atom Coordinates [Key by (Chain-idx, atom-idx, 3-dimension)]
def _ProteinOnlyFeatures(pdb_model):
  # Keyed by Chain-Idx
  chain_ids = []
  # Keyed by (Chain-idx, atom-idx)
  atoms = []

  # Used to compute the center of mass.
  center_of_mass = np.zeros(3)
  total_num_atoms = 0

  for c in pdb_model.get_chains():
    chain_ids.append(c.id)
    residue_seqid =0
    # Keyed by atom-idx
    chain_atoms = {
        'residue_seqid': [],
        'resname': [],
        'hetflag': [],
        'resseq': [],
        'icode': [],
        'atom_name': [],
        'atom_coords': []
    }
    for r in c.get_residues():
      if not PDB.Polypeptide.is_aa(r):
        # Ignore this residue. It's not an amino acid.
        continue
      residue_seqid += 1
      resname = r.get_resname()
      hetflag, resseq, icode = r.get_id()
      residue_id = '{}-{}-{}'.format(hetflag, resseq, icode)
      for a in r.get_atoms():
        atom_name = a.get_name()
        atom_coords = a.get_coord()
        chain_atoms['residue_seqid'].append(residue_seqid)
        chain_atoms['resname'].append(resname)
        chain_atoms['hetflag'].append(hetflag)
        chain_atoms['resseq'].append(resseq)
        chain_atoms['icode'].append(icode)
        chain_atoms['atom_name'].append(atom_name)
        chain_atoms['atom_coords'].append(atom_coords)
        center_of_mass += atom_coords
        total_num_atoms += 1
    atoms.append(chain_atoms)

  if total_num_atoms==0:
    return None

  return {
      'structure_id': pdb_model.get_full_id()[0],
      'chain_ids': tf.constant(chain_ids),
      'residue_seqid': tf.ragged.constant([ca['residue_seqid'] for ca in atoms]),
      'resname': tf.ragged.constant([ca['resname'] for ca in atoms]),
      'hetflag': tf.ragged.constant([ca['hetflag'] for ca in atoms]),
      'resseq': tf.ragged.constant([ca['resseq'] for ca in atoms]),
      'icode': tf.ragged.constant([ca['icode'] for ca in atoms]),
      'atom_name': tf.ragged.constant([ca['atom_name'] for ca in atoms]),
      'atom_coords': tf.ragged.constant(
          [np.array(ca['atom_coords']) - center_of_mass/total_num_atoms for ca in atoms],
          dtype=tf.float32),
  }

def _BytesFeature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _BytesListFeature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def SerializeRaggedTensor(rt):
  components = tf.nest.flatten(rt, expand_composites=True)
  return tf.stack([tf.io.serialize_tensor(t) for t in components])

def DeserializeRaggedTensor(serialized, type_spec):
  component_specs = tf.nest.flatten(type_spec, expand_composites=True)
  components = [
      tf.io.parse_tensor(serialized[i], spec.dtype)
      for i, spec in enumerate(component_specs)]
  return tf.nest.pack_sequence_as(type_spec, components, expand_composites=True)

def ProteinOnlyExamples(pdb_structure):
  for m in pdb_structure.get_models():
    features = _ProteinOnlyFeatures(m)
    yield {
        'structure_id': _BytesFeature(bytes(features['structure_id'], 'utf-8')),
        'chain_ids': _BytesFeature(
          tf.io.serialize_tensor(features['chain_ids']).numpy()),
        'residue_seqid': _BytesListFeature(
          SerializeRaggedTensor(features['residue_seqid']).numpy()),
        'resname': _BytesListFeature(
          SerializeRaggedTensor(features['resname']).numpy()),
        'hetflag': _BytesListFeature(
          SerializeRaggedTensor(features['hetflag']).numpy()),
        'resseq': _BytesListFeature(
          SerializeRaggedTensor(features['resseq']).numpy()),
        'icode': _BytesListFeature(
          SerializeRaggedTensor(features['icode']).numpy()),
        'atom_name': _BytesListFeature(
          SerializeRaggedTensor(features['atom_name']).numpy()),
        'atom_coords': _BytesListFeature(
          SerializeRaggedTensor(features['atom_coords']).numpy()),
        }

def SimpleExample(example):
    feature  ={'name': _BytesFeature(bytes(example['name'], 'utf-8')),
            'residue_names': _BytesFeature(
                tf.io.serialize_tensor(tf.constant(example['residue_names'])).numpy()),
            'atom_names': _BytesFeature(
                tf.io.serialize_tensor(tf.constant(example['atom_names'])).numpy()),
            'normalized_coordinates': _BytesFeature(
                tf.io.serialize_tensor(tf.constant(example['normalized_coordinates'])).numpy())
            }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

class GetTrainingSummariesFn(beam.CombineFn):
    def create_accumulator(self):
        return (set(),set())
    
    def add_input(self, accumulator, input):
      accumulator[0].update(input['residue_names'])
      accumulator[1].update(input['atom_names'])
      return accumulator

    def merge_accumulators(self, accumulators):
        return(set.union(
            *(accumulator[0] for accumulator in accumulators)),
            set.union(
                *(accumulator[1] for accumulator in accumulators)))

    def extract_output(self, accumulator):
        return {'residue_names': list(accumulator[0]),
                'atom_names': list(accumulator[1])}
