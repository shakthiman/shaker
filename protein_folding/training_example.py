import apache_beam as beam
import collections
import numpy as np
import io
import random
from sklearn import neighbors

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

def _BytesFeature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

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
