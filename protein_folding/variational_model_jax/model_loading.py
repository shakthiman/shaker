import io

import flax

def _UploadParams(bucket, blob_name, params):
  blob = bucket.blob(blob_name)
  params_byte_string = flax.serialization.to_bytes(params)
  file = io.BytesIO(params_byte_string)
  blob.upload_from_file(file)


def SaveModel(storage_client, bucket_name, blob_name, encoder_params, conditioner_params, decoder_params):
  bucket = storage_client.bucket(bucket_name)
  _UploadParams(bucket, blob_name + "/encoder", encoder_params)
  _UploadParams(bucket, blob_name + "/conditioner", conditioner_params)
  _UploadParams(bucket, blob_name + "/decoder", decoder_params)
