import apache_beam as beam
import collections
import numpy as np
import io
import random
from sklearn import neighbors

import tensorflow as tf

# Config for Example Generation
ExampleConfig = collections.namedtuple('ExampleConfig',
        ['max_num_atoms', 'num_nearby_atoms', 'beta', 'maxT'])

def TrainingExampleSchema():
    return {
        "namespace": "ml_fun.training_example",
        "type": "record",
        "name": "TrainingExample",
        "fields": [
            {"name": "name", "type": "string"},
            {"name": "residue_names", "type": "bytes"},
            {"name": "atom_names", "type": "bytes"},
            {"name": "normalized_coordinates", "type": "bytes"}
        ]
    }

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

def _numpyToBytes(arr):
    serialized_numpy = io.BytesIO()
    np.save(serialized_numpy, arr)
    return serialized_numpy.getvalue()

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

def OptimizeExample(example):
    return {
        'name': example['name'],
        'residue_names': _numpyToBytes(example['residue_names']),
        'atom_names': _numpyToBytes(example['atom_names']),
        'normalized_coordinates': _numpyToBytes(example['normalized_coordinates'])
    }

def _IntFeature(value):
    return tf.train.Feature(int64_list=tf.train.BytesList(value=[value]))

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

class TFRecordDoFn(beam.DoFn):
    def __init__(self, shared_handle, example_config):
        self._shared_handle = shared_handle
        self._example_config = example_config

    def process(self, preprocessed_pdb, example_summary):
        # Fetch Information about the PDBs.
        residue_names = preprocessed_pdb['residue_names']
        atom_names = preprocessed_pdb['atom_names']
        normalized_coordinates = preprocessed_pdb['normalized_coordinates']

        # Add noise to the PDB.
        t = random.randint(1, self._example_config.maxT)
        epsilon = np.random.randn(
                normalized_coordinates.shape[0],
                normalized_coordinates.shape[1])
        alpha = 1 - self._example_config.beta
        alpha_tilde = pow(alpha, t)
        preturbed_coordinates = (
                pow(alpha_tilde, 0.5) * normalized_coordinates
                + pow(1-alpha_tilde, 0.5)*epsilon)

        # For each atom find nearby atoms.
        n_neighbors=min(self._example_config.num_nearby_atoms+1,
                len(preturbed_coordinates))
        nbrs = neighbors.NearestNeighbors(
                n_neighbors=n_neighbors).fit(preturbed_coordinates)
        _, indices = nbrs.kneighbors(preturbed_coordinates)

        nearest_coordinates = preturbed_coordinates[indices[:,1:]]
        nearest_atoms_names = atom_names[indices[:,1:]]
        nearest_residue_names = residue_names[indices[:,1:]]

        # Convert the residues/atoms using the vocabulary.
        def _GetStaticVocab(vocab):
            return tf.lookup.StaticVocabularyTable(
                    tf.lookup.KeyValueTensorInitializer(
                        tf.constant(vocab),
                        tf.constant(range(2, len(vocab)+2), dtype=tf.int64),
                        key_dtype=tf.string,
                        value_dtype=tf.int64), 1)
        residue_names_preprocessor = self._shared_handle.acquire(
                lambda: _GetStaticVocab(example_summary['residue_names']),
                'residue_names_preprocessor')
        atom_names_preprocessor = self._shared_handle.acquire(
                lambda: _GetStaticVocab(example_summary['atom_names']),
                'atom_names_preprocessor')

        residue_names = residue_names_preprocessor.lookup(
                tf.constant(preprocessed_pdb['residue_names'], dtype=tf.string))
        nearest_residue_names = residue_names_preprocessor.lookup(
                tf.constant(nearest_residue_names, dtype=tf.string))
        atom_names = atom_names_preprocessor.lookup(
                tf.constant(preprocessed_pdb['atom_names'], dtype=tf.string))
        nearest_atoms_names = atom_names_preprocessor.lookup(
                tf.constant(nearest_atoms_names, dtype=tf.string))

        # Pad the nearest* features.
        nearest_coordinates = np.pad(
                nearest_coordinates, [
                    (0,0),
                    (0,self._example_config.num_nearby_atoms-n_neighbors+1),
                    (0,0)],
                constant_values=[(0,0), (0,0), (0,0)])
        nearest_atoms_names = np.pad(
                nearest_atoms_names,[
                    (0,0),
                    (0, self._example_config.num_nearby_atoms-n_neighbors+1)],
                constant_values=[(0,0),(0,0)])
        nearest_residue_names = np.pad(
                nearest_residue_names,[
                    (0,0),
                    (0, self._example_config.num_nearby_atoms-n_neighbors+1)],
                constant_values=[(0,0),(0,0)])
        nearest_indices = np.pad(
                indices[:,1:], [
                    (0,0),
                    (0, self._example_config.num_nearby_atoms-n_neighbors+1)],
                constant_values=[(-1,-1),(-1,-1)])

        # Pad the Features so they have the expected length.
        feature = {
                'name': _BytesFeature(bytes(preprocessed_pdb['name'], 'utf-8')),
                'residue_names': _BytesFeature(tf.io.serialize_tensor(
                    tf.constant(tf.keras.utils.pad_sequences([residue_names],
                        self._example_config.max_num_atoms)[0])).numpy()),
                'atom_names': _BytesFeature(tf.io.serialize_tensor(
                    tf.constant(tf.keras.utils.pad_sequences([atom_names],
                        self._example_config.max_num_atoms)[0])).numpy()),
                'normalized_coordinates': _BytesFeature(tf.io.serialize_tensor(
                    tf.constant(tf.keras.utils.pad_sequences([normalized_coordinates],
                        self._example_config.max_num_atoms, 'float64')[0])).numpy()),
                'preturbed_coordinates': _BytesFeature(tf.io.serialize_tensor(
                    tf.constant(tf.keras.utils.pad_sequences([preturbed_coordinates],
                        self._example_config.max_num_atoms, 'float64')[0])).numpy()),
                'epsilon': _BytesFeature(tf.io.serialize_tensor(
                    tf.constant(tf.keras.utils.pad_sequences([epsilon],
                        self._example_config.max_num_atoms, 'float64')[0])).numpy()),
                'nearest_atoms_names': _BytesFeature(tf.io.serialize_tensor(
                    tf.constant(tf.keras.utils.pad_sequences([nearest_atoms_names],
                        self._example_config.max_num_atoms)[0])).numpy()),
                'nearest_indices': _BytesFeature(tf.io.serialize_tensor(
                    tf.constant(tf.keras.utils.pad_sequences([nearest_atoms_names],
                        self._example_config.max_num_atoms, value=-1)[0])).numpy()),
                'nearest_residue_names': _BytesFeature(tf.io.serialize_tensor(
                    tf.constant(tf.keras.utils.pad_sequences([nearest_residue_names],
                        self._example_config.max_num_atoms)[0])).numpy()),
                'nearest_coordinates': _BytesFeature(tf.io.serialize_tensor(
                    tf.constant(tf.keras.utils.pad_sequences([nearest_coordinates],
                        self._example_config.max_num_atoms, 'float64')[0])).numpy()),
                }
        yield tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()

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
