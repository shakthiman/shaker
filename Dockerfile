FROM apache/beam_python3.9_sdk:2.52.0 AS base

COPY requirements.in ./

RUN pip install pip-tools
# Install BioPython.
RUN pip-compile --resolver backtracking -o requirements.txt requirements.in
RUN pip-sync requirements.txt

# Verify that the image does not have conflicting dependencies.
RUN pip check

FROM base
COPY protein_folding/ ./protein_folding
# Set the entrypoint to Apache Beam SDK launcher.
ENTRYPOINT ["/opt/apache/beam/boot"]
