from google.cloud import storage
import tensorflow as tf
import random

from protein_folding import training_example

def _IgnoreCondition(x):
  peptide_shapes = tf.map_fn(lambda y: tf.shape(y)[0], x['resname'], fn_output_signature=tf.int32)
  num_peptides = tf.shape(peptide_shapes)[0]
  atoms_per_peptide = tf.math.reduce_max(peptide_shapes)
  return tf.math.reduce_any(
          tf.stack([
            # Hack to ignore sequences with RNA.
            tf.math.equal(tf.math.reduce_min(peptide_shapes), 0),
            # Too many atoms
            tf.math.greater(num_peptides * atoms_per_peptide, 32768)]))

def _PrepareTFDataset(filenames, files_to_take=1):
  return (
      tf.data.TFRecordDataset(tf.random.shuffle(filenames)[0:files_to_take])
      .map(training_example.ParseProteinOnlyExample)
      .filter(lambda x: not _IgnoreCondition(x)))

def _CreateInterleavedDataset(files_by_cluster, num_parallel_calls,
    cluster_shuffle_size, cluster_cycle_length):
  all_files = [tf.io.serialize_tensor(tf.constant(files)) for cluster, files in files_by_cluster.items()]
  return (
      tf.data.Dataset.from_tensor_slices(all_files)
      .repeat()
      .shuffle(cluster_shuffle_size)
      .interleave(
        lambda filenames: _PrepareTFDataset(tf.io.parse_tensor(filenames, tf.string)),
        num_parallel_calls=num_parallel_calls,
        cycle_length=cluster_cycle_length))

def GetTFExamples(project, bucket, blob_prefix, num_parallel_calls,
    cluster_shuffle_size, cluster_cycle_length):
  client = storage.Client(project)
  blobs = client.list_blobs(bucket, prefix=blob_prefix)
  files_by_cluster = dict()
  all_prefixes = set()
  for b in blobs:
    prefix, cluster_id, suffix = b.name.rsplit('/', 2)
    files_by_cluster.setdefault(cluster_id, []).append(
        '/'.join(["gs:/", bucket, prefix, cluster_id, suffix]))
    all_prefixes.add(prefix)
  assert len(all_prefixes)==1, "Should have only 1 prefix."
  return _CreateInterleavedDataset(files_by_cluster, num_parallel_calls,
      cluster_shuffle_size, cluster_cycle_length)

def GetSmallTFExamples(project, bucket, blob_prefix, size):
  client = storage.Client(project)
  blobs = client.list_blobs(bucket, prefix=blob_prefix)
  files_by_cluster = dict()
  all_prefixes = set()
  for b in blobs:
    prefix, cluster_id, suffix = b.name.rsplit('/', 2)
    files_by_cluster.setdefault(cluster_id, []).append(
        '/'.join(["gs:/", bucket, prefix, cluster_id, suffix]))
    all_prefixes.add(prefix)
  assert len(all_prefixes)==1, "Should have only 1 prefix."

  raw_datasets = []
  for cluster, files in list(files_by_cluster.items())[:size]:
    raw_datasets.append(_PrepareTFDataset(files))

  return tf.data.Dataset.sample_from_datasets(raw_datasets)
