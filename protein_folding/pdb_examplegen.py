from apache_beam.io import fileio
import apache_beam as beam
import logging
import struct

from google.cloud import storage
from Bio import PDB
from protein_folding import training_example
from protein_folding import protein_cluster_lookup
from protein_folding import ligand_check
from apache_beam.io import tfrecordio

_PDB_DOWNLOAD_BUCKET_NAME = 'rcsb_download'
_CLUSTERED_PDB_BUCKET_NAME = 'clustered_pdb'
_SHAKER_DOWNLOADS = 'shaker_downloads'
_NO_CLUSTER_TAG = 'no_cluster'
_POLYPEPTIDE_TAG = 'polypeptides'
_SINGLE_CHAIN_TAG = 'single_chain'

def _GetPDBStructure(pdb_id):
  storage_client = storage.Client()
  blob = storage_client.bucket(_PDB_DOWNLOAD_BUCKET_NAME).blob('{}.cif'.format(pdb_id))
  parser = PDB.FastMMCIFParser()
  return parser.get_structure(pdb_id, blob.open())

def _GetStructurePartition(model, pcl, lch):
  # Collect information required to know the protein's partition.
  # Whether this model just has a single chain
  is_single_chain = (len(list(model.get_chains())) == 1)
  # The Protein's Id in the PDB Database
  pid = model.get_full_id()[0]
  all_clusters = pcl.ProteinCluster(pid)
  if all_clusters is None:
    all_clusters = [_NO_CLUSTER_TAG]
  has_ligand = lch.HasLigands(pid)

  # We do not currently handle cases where the structure has a ligand.
  if has_ligand:
    return []

  all_partitions = []
  for c in all_clusters:
    all_partitions.append("{}/{}".format(_POLYPEPTIDE_TAG, c))

  if is_single_chain:
    for c in all_clusters:
      all_partitions.append("{}/{}".format(_SINGLE_CHAIN_TAG, c))

  return all_partitions

def _ReadToTrainingFeatures(pdb_id, pcl, lch):
  # Short circuit and do not read ligands.
  if lch.HasLigands(pdb_id):
    return []
  try:
    structure = _GetPDBStructure(pdb_id)
  except Exception as e:
    logging.warning(
        'Constructing PDB Structure failed with: {}'.format(str(e)))
    return []

  for m in structure.get_models():
    partitions = _GetStructurePartition(m,pcl, lch)
    for p in partitions:
      e = training_example.ProteinOnlyFeatures(m)
      if e is not None:
        yield (p, e)

class TrainingFeaturesDoFn(beam.DoFn):
  def __init__(self, lig_pairs_blob, clusters_by_entity_blob):
    self._lig_pairs_blob = lig_pairs_blob
    self._clusters_by_entity_blob = clusters_by_entity_blob

  def setup(self):
    storage_client = storage.Client()
    self._pcl = protein_cluster_lookup.ProteinClusterLookup(
        storage_client.bucket(_CLUSTERED_PDB_BUCKET_NAME)
        .blob(self._clusters_by_entity_blob).open())
    self._lch = ligand_check.LigandCheck(
        storage_client.bucket(_SHAKER_DOWNLOADS)
        .blob(self._lig_pairs_blob).open())

  def process(self, pdb_id):
    for e in _ReadToTrainingFeatures(pdb_id, self._pcl, self._lch):
      yield e

class TFExampleSink(fileio.FileSink):

  def open(self, fh):
    self._fh = fh

  def write(self, record):
    tfrecordio._TFRecordUtil.write_record(self._fh, record[1])

  def flush(self):
    pass
  
def DownloadTrainingExamples(pdb_ids, target_location, summary_location, runner, options):
  storage_client = storage.Client()
  clusters_by_entity = 'clusters-by-entity-40.txt'
  pcl = protein_cluster_lookup.ProteinClusterLookup(
      storage_client.bucket(_CLUSTERED_PDB_BUCKET_NAME)
      .blob(clusters_by_entity).open())
  all_cluster_values = set()
  for pid in pdb_ids:
    clusters = pcl.ProteinCluster(pid)
    if clusters is None:
      all_cluster_values.add(_NO_CLUSTER_TAG)
      continue
    all_cluster_values.update(["{}".format(c) for c in clusters])

  all_tags = ["{}/{}".format(_POLYPEPTIDE_TAG, c) for c in all_cluster_values]
  all_tags = all_tags + ["{}/{}".format(_SINGLE_CHAIN_TAG, c) for c in all_cluster_values]

  p = beam.Pipeline()
  training_features= (
      p | 'Create initial values' >> beam.Create(pdb_ids)
        | 'Retrieve Examples' >> beam.ParDo(TrainingFeaturesDoFn(
            'lig_pairs.lst', clusters_by_entity)))
  # Compute the Summaries
  (training_features
    | 'Remove Partition' >> beam.Map(lambda x: x[1])
    | 'Summarize Training Examples' >> beam.CombineGlobally(
        training_example.GetProteinOnlyTrainingSummariesFn())
    | 'WriteToAvro' >> beam.io.WriteToAvro(
        summary_location,
        training_example.TrainingSummarySchema(),
        file_name_suffix=".avro"))

  destination_fn = lambda t: '{}/data'.format(t)
  (training_features
    | 'Optimize Example for Training' >> beam.Map(
        lambda x: (x[0], training_example.ProteinOnlyExample(x[1])))
    | 'WriteExamplesToTFRecord' >> fileio.WriteToFiles(
        path=target_location,
        destination=lambda x: destination_fn(x[0]),
        sink=lambda d: TFExampleSink(),
        shards=500,
        file_naming=fileio.destination_prefix_naming('.tfrecord')))
  runner.run_pipeline(p, options)
