from avro import datafile
from avro import io

import tensorflow as tf

def _GetStaticVocab(vocab):
  return tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(
        tf.constant(vocab),
        tf.constant(range(2, len(vocab)+2), dtype=tf.int64),
        key_dtype=tf.string,
        value_dtype=tf.int64), 1)

class PDBVocab(object):
  def __init__(self, training_summaries_file):
    reader = datafile.DataFileReader(
        training_summaries_file.open('rb'), io.DatumReader())
    row = next(reader)
    residue_names = row['residue_names']
    atom_names = row['atom_names']
    
    self._residue_names_lookup = _GetStaticVocab(residue_names)
    self._atom_names_lookup = _GetStaticVocab(atom_names)

    self._residue_lookup_size = len(residue_names) + 2
    self._atom_lookup_size = len(atom_names) + 2

  def GetResidueNamesId(self, residue_names):
    return self._residue_names_lookup.lookup(residue_names)

  def ResidueLookupSize(self):
    return self._residue_lookup_size

  def GetAtomNamesId(self, atom_names):
    return self._atom_names_lookup.lookup(atom_names)

  def AtomLookupSize(self):
    return self._atom_lookup_size
