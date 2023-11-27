FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app
COPY src/ /usr/src/app/src/
COPY scripts/ /usr/src/app/scripts/
COPY requirements.txt /usr/src/app/
COPY setup.py /usr/src/app/

# Install the colour_som package
RUN pip install .

# Copy the entrypoint script into the container and set it as entrypoint
COPY ./scripts/entrypoint.sh /usr/src/app/
RUN chmod +x /usr/src/app/entrypoint.sh
RUN ls /usr/src/app

ENTRYPOINT ["/usr/src/app/entrypoint.sh"]
