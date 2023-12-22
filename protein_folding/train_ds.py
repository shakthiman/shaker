from google.cloud import storage
import tensorflow as tf

from protein_folding import training_example

def GetTFExamples(project, bucket, blob_prefix):
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
    raw_datasets.append(tf.data.TFRecordDataset(files).repeat())

  return tf.data.Dataset.sample_from_datasets(raw_datasets).map(training_example.ParseProteinOnlyExample)
