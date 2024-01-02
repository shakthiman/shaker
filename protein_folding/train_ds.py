from google.cloud import storage
import tensorflow as tf

from protein_folding import training_example

# Hack to ignore sequences with RNA.
def _IgnoreCondition(x):
  peptide_shapes = tf.map_fn(lambda y: tf.shape(y)[0], x['resname'], fn_output_signature=tf.int32)
  return tf.math.reduce_min(peptide_shapes)==0

def _PrepareTFDataset(tf_dataset, num_parallel_calls):
  return (tf_dataset.repeat()
      .map(training_example.ParseProteinOnlyExample,
          num_parallel_calls=num_parallel_calls)
      .filter(lambda x: not _IgnoreCondition(x)))

def GetTFExamples(project, bucket, blob_prefix, num_parallel_calls):
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
  for cluster, files in files_by_cluster.items():
    raw_datasets.append(_PrepareTFDataset(
      tf.data.TFRecordDataset(files), num_parallel_calls))

  return tf.data.Dataset.sample_from_datasets(raw_datasets)

def GetSmallTFExamples(project, bucket, blob_prefix, num_parallel_calls, size):
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
    raw_datasets.append(_PrepareTFDataset(
      tf.data.TFRecordDataset(files), num_parallel_calls))

  return tf.data.Dataset.sample_from_datasets(raw_datasets)
