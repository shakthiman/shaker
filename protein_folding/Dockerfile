FROM apache/beam_python3.9_sdk:2.46.0 AS base

# Install BioPython.
RUN pip install --no-cache-dir biopython
RUN pip install --no-cache-dir google-cloud-storage
RUN pip install --no-cache-dir scikit-learn

# Verify that the image does not have conflicting dependencies.
RUN pip check

FROM base
COPY *.py ./
# Set the entrypoint to Apache Beam SDK launcher.
ENTRYPOINT ["/opt/apache/beam/boot"]
