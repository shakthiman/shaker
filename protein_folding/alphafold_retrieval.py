import apache_beam as beam
from apache_beam.io.gcp import gcsio
from apache_beam.io.gcp import gcsfilesystem
from apache_beam.io import filesystem
from apache_beam.io import textio
from apache_beam.utils import shared
from google.cloud import storage
import protein_structure
from Bio import PDB
from Bio.PDB import MMCIFParser
import io
import protein_structure
import random
import training_example

import tensorflow as tf

_ALPHAFOLD_BUCKET_NAME = 'public-datasets-deepmind-alphafold-v4'

def _GetPDBStructure(entry_name):
    storage_client = storage.Client()
    blob = storage_client.bucket(_ALPHAFOLD_BUCKET_NAME).blob(entry_name + "-model_v4.cif")
    parser = PDB.FastMMCIFParser()
    return parser.get_structure(entry_name, blob.open())

def _ReadToInternalPDBStructure(options, entry_name):
    return protein_structure.StructureFromPDBStructure(
        entry_name, _GetPDBStructure(entry_name))

def _ReadToTrainingExample(entry_name):
    return training_example.PreProcessPDBStructure(
        _GetPDBStructure(entry_name))

def DownloadProteinStructures(entry_names, target_location, runner, options):
  p = beam.Pipeline()
  outputs = (p
        | 'Create initial values' >> beam.Create(entry_names)
        | 'Retrieve Structure' >> beam.Map(lambda x: _ReadToInternalPDBStructure(options, x))
        | 'Convert the Structure to Avro' >> beam.Map(lambda x: protein_structure.ConvertStructureToAvro(x))
        | 'WriteToAvro' >> beam.io.WriteToAvro(
            target_location,
            protein_structure.AllAvroSchemas(),
             file_name_suffix=".avro"))
  runner.run_pipeline(p, options)

def DownloadTrainingExamples(entry_names, target_location, summary_location, runner, options):
  p = beam.Pipeline()
  training_examples = (p
        | 'Create initial values' >> beam.Create(entry_names)
        | 'Retrieve Structure' >> beam.Map(lambda x: _ReadToTrainingExample(x)))

  p1 = (training_examples
          | 'Summarize Training Examples' >> beam.CombineGlobally(
              training_example.GetTrainingSummariesFn())
          | 'WriteSummariesToAvro' >> beam.io.WriteToAvro(
              summary_location,
              training_example.TrainingSummarySchema(),
              file_name_suffix=".avro"))

  p2 = (training_examples
          | 'Optimize Example for Training' >> beam.Map(
              lambda x: training_example.SimpleExample(x))
          | 'WriteExamplesToTFRecord' >> beam.io.tfrecordio.WriteToTFRecord(
              target_location,
              file_name_suffix=".tfrecord"))

  

  runner.run_pipeline(p, options)
