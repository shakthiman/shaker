import alphafold_retrieval
import training_example
import google.auth
from google.cloud import bigquery
from apache_beam import runners
from apache_beam.runners.portability import portable_runner
from apache_beam.options import pipeline_options

def main():
    query = """
    SELECT
      fractionPlddtVeryHigh,
      CONCAT('gs://public-datasets-deepmind-alphafold-v4/', entryID, '-model_v4.cif') AS file,
      entryID,
      uniprotEnd
    FROM `bigquery-public-data.deepmind_alphafold.metadata`
    WHERE organismScientificName='Homo sapiens'
    AND fractionPlddtVeryHigh>0.95
    """
    client = bigquery.Client()
    query_job = client.query(query)
    entries = [row['entryID'] for row in query_job]
    print(entries)
    options = pipeline_options.PipelineOptions()
    _, options.view_as(pipeline_options.GoogleCloudOptions).project = google.auth.default()
    options.view_as(pipeline_options.GoogleCloudOptions).region = 'us-central1'
    dataflow_gcs_location = 'gs://unreplicated-training-data/dataflow'
    options.view_as(pipeline_options.GoogleCloudOptions).staging_location = '%s/staging' % dataflow_gcs_location
    options.view_as(pipeline_options.GoogleCloudOptions).temp_location = '%s/temp' % dataflow_gcs_location
    options.view_as(pipeline_options.GoogleCloudOptions).temp_location = '%s/temp' % dataflow_gcs_location
    options.view_as(pipeline_options.WorkerOptions).max_num_workers = 20
    options.view_as(pipeline_options.WorkerOptions).sdk_container_image="gcr.io/shaker-388116/structure-import:live"

    #alphafold_retrieval.DownloadProteinStructures(
    #    entries,
    #    "gs://high-confidence-protein-files/structures2/data",
    #    runners.DataflowRunner(), options)
    alphafold_retrieval.DownloadTrainingExamples(
        entries,
        "gs://unreplicated-training-data/training_examples/data",
        "gs://unreplicated-training-data/training_examples_summary/data", runners.DataflowRunner(), options)

if __name__=="__main__":
    main()
