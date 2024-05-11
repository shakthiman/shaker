FROM tensorflow/tensorflow:nightly AS base

COPY assembly_training/requirements.in ./

RUN pip install pip-tools
# Install BioPython.
RUN pip-compile --resolver backtracking -o requirements.txt requirements.in
RUN pip-sync requirements.txt

RUN wget https://storage.googleapis.com/cloud-tpu-tpuvm-artifacts/libtpu/libtpu.so -O /lib/libtpu.so
RUN chmod 777 /lib/libtpu.so

# Verify that the image does not have conflicting dependencies.
RUN pip check

FROM base
COPY protein_folding/ ./protein_folding
COPY assembly_training/thailam.py ./
# Set the entrypoint to Apache Beam SDK launcher.
ENTRYPOINT ["python3", "thailam.py"]
