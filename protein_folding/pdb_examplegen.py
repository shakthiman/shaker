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
_ASSEMBLIES_DOWNLOAD_BUCKET_NAME = 'rcsb_assemblies_download'
_CLUSTERED_PDB_BUCKET_NAME = 'clustered_pdb'
_SHAKER_DOWNLOADS = 'shaker_downloads'
_NO_CLUSTER_TAG = 'no_cluster'
_POLYPEPTIDE_TAG = 'polypeptides'
_SINGLE_CHAIN_TAG = 'single_chain'

def _GetPDBStructure(pdb_id):
  storage_client = storage.Client()
  structure_blob = storage_client.bucket(_PDB_DOWNLOAD_BUCKET_NAME).blob(
      '{}.cif'.format(pdb_id))
  parser = PDB.MMCIFParser()

  if structure_blob.exists():
    return parser.get_structure(pdb_id, structure_blob.open())

  return None

def _GetAssemblyStructure(assembly_blob_name):
  storage_client = storage.Client()
  assembly_blob = storage_client.bucket(
      _ASSEMBLIES_DOWNLOAD_BUCKET_NAME).get_blob(assembly_blob_name)
  if assembly_blob.size > 5e8:
    return None

  fast_parser = PDB.FastMMCIFParser()
  return fast_parser.get_structure(
      assembly_blob_name.removesuffix(".cif"),
      assembly_blob.open())

def _GetStructurePartition(model, pdb_id, pcl, lch):
  # Collect information required to know the protein's partition.
  # Whether this model just has a single chain
  is_single_chain = (len(list(model.get_chains())) == 1)
  # The Protein's Id in the PDB Database
  all_clusters = pcl.ProteinCluster(pdb_id)
  if all_clusters is None:
    all_clusters = [_NO_CLUSTER_TAG]
  has_ligand = lch.HasLigands(pdb_id)

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

def _YieldRelevantAssemblyBlobNames(pdb_id, assembly_blob_names, lch, resolution_threshold):
  # Short circuit and do not read ligands.
  if lch.HasLigands(pdb_id):
    return []
  try:
    structure = _GetPDBStructure(pdb_id)
    if (structure is not None and
        ('resolution' in structure.header) and
        (structure.header['resolution'] is not None) and
        (structure.header['resolution']>resolution_threshold)):
      # The structure is not reliable.
      return []
  except Exception as e:
    logging.warning(
        'Constructing PDB Structure failed with: {}'.format(str(e)))
    return []

  for assn in assembly_blob_names:
    yield (pdb_id, assn)


def _ReadToTrainingFeatures(pdb_id, assembly_blob_name, pcl, lch):
  try:
    assembly_structure = _GetAssemblyStructure(assembly_blob_name)
  except Exception as e:
    logging.warning(
        'Constructing PDB Assembly failed with: {}'.format(str(e)))
    return []

  if assembly_structure is None:
    return []

  for m in assembly_structure.get_models():
    partitions = _GetStructurePartition(m, pdb_id, pcl, lch)
    e = training_example.ProteinOnlyFeatures(m)
    for p in partitions:
      if e is not None:
        yield (p, e)

class TrainingFeaturesDoFn(beam.DoFn):
  def __init__(
      self, lig_pairs_blob, clusters_by_entity_blob):
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

  def process(self, pid_assembly_blob_name):
    pdb_id = pid_assembly_blob_name[0]
    assembly_blob_name = pid_assembly_blob_name[1]
    for e in _ReadToTrainingFeatures(
        pdb_id, assembly_blob_name, self._pcl, self._lch):
      yield e

class RelevantAssembliesDoFn(beam.DoFn):
  def __init__(
      self, lig_pairs_blob, resolution_threshold):
    self._lig_pairs_blob = lig_pairs_blob
    self._resolution_threshold = resolution_threshold

  def setup(self):
    storage_client = storage.Client()
    self._lch = ligand_check.LigandCheck(
        storage_client.bucket(_SHAKER_DOWNLOADS)
        .blob(self._lig_pairs_blob).open())
  
  def process(self, pid_assembly_blob_names):
    pdb_id = pid_assembly_blob_names[0]
    assembly_blob_names = pid_assembly_blob_names[1]
    for pdb_id_assn in _YieldRelevantAssemblyBlobNames(
        pdb_id, assembly_blob_names, self._lch, self._resolution_threshold):
      yield pdb_id_assn

class TFExampleSink(fileio.FileSink):

  def open(self, fh):
    self._fh = fh

  def write(self, record):
    tfrecordio._TFRecordUtil.write_record(self._fh, record[1])

  def flush(self):
    pass
  
def DownloadTrainingExamples(pid_assemblies, target_location, summary_location, runner, options,
    resolution_threshold=9.0):
  storage_client = storage.Client()
  clusters_by_entity = 'clusters-by-entity-40.txt'
  pcl = protein_cluster_lookup.ProteinClusterLookup(
      storage_client.bucket(_CLUSTERED_PDB_BUCKET_NAME)
      .blob(clusters_by_entity).open())

  p = beam.Pipeline()
  training_features= (
      p | 'Create initial values' >> beam.Create(pid_assemblies)
        | 'Filter to Relevant Assemblies' >> beam.ParDo(RelevantAssembliesDoFn(
          lig_pairs_blob='lig_pairs.lst',
          resolution_threshold=resolution_threshold))
        | 'Rebalance Data' >> beam.Reshuffle()
        | 'Retrieve Examples' >> beam.ParDo(TrainingFeaturesDoFn(
            lig_pairs_blob='lig_pairs.lst',
            clusters_by_entity_blob=clusters_by_entity)))
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
