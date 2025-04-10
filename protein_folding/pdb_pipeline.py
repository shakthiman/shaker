from protein_folding import pdb_examplegen
import google.auth
from google.cloud import storage
from apache_beam import runners
from apache_beam.runners.portability import portable_runner
from apache_beam.options import pipeline_options

def main():
    client = storage.Client()
    pid_assemblies = dict()
    for b in client.bucket("rcsb_assemblies_download").list_blobs():
      pid_assemblies.setdefault(b.name.split("-")[0], []).append(b.name)
    options = pipeline_options.PipelineOptions()
    _, options.view_as(pipeline_options.GoogleCloudOptions).project = google.auth.default()
    options.view_as(pipeline_options.GoogleCloudOptions).region = 'us-central1'
    dataflow_gcs_location = 'gs://unreplicated-training-data/dataflow'
    options.view_as(pipeline_options.GoogleCloudOptions).staging_location = '%s/staging' % dataflow_gcs_location
    options.view_as(pipeline_options.GoogleCloudOptions).temp_location = '%s/temp' % dataflow_gcs_location
    options.view_as(pipeline_options.GoogleCloudOptions).temp_location = '%s/temp' % dataflow_gcs_location
    options.view_as(pipeline_options.DebugOptions).add_experiment('shuffle_mode=service')
    options.view_as(pipeline_options.WorkerOptions).max_num_workers = 20
    options.view_as(pipeline_options.WorkerOptions).sdk_container_image="gcr.io/shaker-388116/structure-import:live"
    options.view_as(pipeline_options.WorkerOptions).machine_type="e2-highmem-4"

    pdb_examplegen.DownloadTrainingExamples(
        pid_assemblies,
        "gs://unreplicated-training-data/pdb_training_examples_mar_26",
        "gs://unreplicated-training-data/pdb_training_examples_summary/data_mar_26", runners.DataflowRunner(), options)

if __name__=="__main__":
    main()
