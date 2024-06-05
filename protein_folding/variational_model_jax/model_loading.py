import io

import flax

def _UploadParams(bucket, blob_name, params):
  blob = bucket.blob(blob_name)
  params_byte_string = flax.serialization.to_bytes(params)
  file = io.BytesIO(params_byte_string)
  blob.upload_from_file(file)

def _DownloadParams(bucket, blob_name, template_params):
  blob = bucket.blob(blob_name)
  params_byte_string = blob.download_as_bytes()
  return flax.serialization.from_bytes(template_params, params_byte_string)

def SaveModel(storage_client, bucket_name, blob_name, encoder_params, conditioner_params, decoder_params, opt_state):
  bucket = storage_client.bucket(bucket_name)
  _UploadParams(bucket, blob_name + "/encoder", encoder_params)
  _UploadParams(bucket, blob_name + "/conditioner", conditioner_params)
  _UploadParams(bucket, blob_name + "/decoder", decoder_params)
  _UploadParams(bucket, blob_name + "/opt_state", opt_state)

def LoadModel(storage_client, bucket_name, blob_name, encoder_params, conditioner_params, decoder_params):
  bucket = storage_client.bucket(bucket_name)
  return (
      _DownloadParams(bucket, blob_name + '/encoder', encoder_params),
      _DownloadParams(bucket, blob_name + '/conditioner', conditioner_params),
      _DownloadParams(bucket, blob_name + '/decoder', decoder_params))

def LoadOptimizer(storage_client, bucket_name, blob_name, opt_state):
  bucket = storage_client.bucket(bucket_name)
  return _DownloadParams(bucket, blob_name + '/opt_state', opt_state)
