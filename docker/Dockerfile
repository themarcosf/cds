### Environment setup #########################################################
FROM mcr.microsoft.com/devcontainers/python:3.13-bookworm

# set PYTHONPATH to include src directory
ENV PYTHONPATH=src

# prevent creation of __pycache__ directories
ENV PYTHONDONTWRITEBYTECODE=1

# make output unbuffered (useful for logging/debugging)
ENV PYTHONUNBUFFERED=1

# install pip-tools
RUN python -m pip install --upgrade pip pip-tools