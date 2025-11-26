FROM ubuntu:latest

# Set environment variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

# Update and install dependencies
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y python3-pip  python3-venv \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-base-apps \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gir1.2-gst-plugins-bad-1.0 python3-gst-1.0 libcairo2 libcairo2-dev \
    git gstreamer1.0-python3-plugin-loader


# Set some environment variables
ENV GST_PLUGIN_PATH=/root/gst-python-ml/plugins

# allow Python to properly handle Unicode characters during logging.
ENV PYTHONIOENCODING=utf-8

COPY pyproject.toml /root/

# Set the working directory (optional)
WORKDIR /root

# Set the entry point to bash for interactive use
CMD ["/bin/bash"]
