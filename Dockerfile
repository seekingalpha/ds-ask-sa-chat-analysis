ARG PYTHON_VERSION=3.9

FROM python:${PYTHON_VERSION}-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /opt/ml/code

# Copy only dependency-related files first
COPY pyproject.toml .

# Install project as a package
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --quiet .

# Copy the application code
COPY . .

# Ensure that the current folder is in the PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:."

RUN echo "Docker pwd" ; pwd
RUN echo "Docker ls" ; ls -lha

# Define the entry point to your script
ENTRYPOINT ["python3"]