# GStreamer Python ML

This project provides a pure Python ML framework for upstream GStreamer, supporting a broad range of ML vision and language features. 

Supported functionality includes:

1. object detection
1. tracking
1. video captioning
1. translation
1. transcription
1. speech to text
1. text to speech
1. text to image
1. LLMs
1. serializing model metadata to Kafka server

Different ML toolkits are supported via the `MLEngine` abstraction - we have nominal support for
TensorFlow, LiteRT and OpenVINO, but all testing thus far has been done with PyTorch.

These elements will work with your distribution's GStreamer packages as long as the GStreamer version
is >= 1.24.

## Install

There are two installation options described below: on host machine or on Docker container:

### Host Install

#### Install distribution packages

##### Ubuntu
```
sudo apt update && sudo apt -y upgrade
sudo apt install -y python3-pip  python3-venv \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-base-apps \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
    gir1.2-gst-plugins-bad-1.0 python3-gst-1.0 gstreamer1.0-python3-plugin-loader \
    libcairo2 libcairo2-dev git
```

##### Fedora

(adjust Fedora version from 42 to match your version number)

```
sudo dnf install https://download1.rpmfusion.org/free/fedora/rpmfusion-free-release-42.noarch.rpm https://download1.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-42.noarch.rpm
sudo dnf update -y
sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda -y
```

```
sudo dnf upgrade -y
sudo dnf install -y python3-pip \
    python3-devel cairo cairo-devel cairo-gobject-devel pkgconfig git \
    gstreamer1-plugins-base gstreamer1-plugins-base-tools \
    gstreamer1-plugins-good gstreamer1-plugins-bad-free \
    gstreamer1-plugins-bad-free-devel python3-gstreamer1
```



#### Manage Python packages with uv

##### install
curl -LsSf https://astral.sh/uv/install.sh | sh

##### set up uv venv

```
uv venv --system-site-packages
source .venv/bin/activate
uv pip install --upgrade pip
uv sync
```

Now manually install flash-attn wheel (must match your version of python, torch and cuda)
For example:

`uv pip install ./flash_attn-2.8.3+cu128torch2.9-cp313-cp313-linux_x86_64.whl`

Pe-built wheels can be found here:
https://github.com/mjun0812/flash-attention-prebuild-wheels/releases


#### Clone repo

```
cd $HOME/src
git clone https://github.com/collabora/gst-python-ml.git
```

#### Update .bashrc

```
echo 'export GST_PLUGIN_PATH=$HOME/src/gst-python-ml/demos:$HOME/src/gst-python-ml/plugins:$GST_PLUGIN_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Docker Install

#### Build Docker Container

Important Note:

This Dockerfile maps a local `gst-python-ml` repository to the container,
and expects this repository to be located in `$HOME/src` i.e.  `$HOME/src/gst-python-ml`.


#### Enable Docker GPU Support on Host

To use the host GPU in a docker container, you will need to install the nvidia container toolkit. If running on CPU, these steps can be skipped.


##### Ubuntu
```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

##### Fedora

```
sudo dnf install docker
sudo usermod -aG docker $USER
# Then either log out/in completely, or:
newgrp docker
```


```
# 1. Add NVIDIA Container Toolkit repository
curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# 2. Remove Fedora's conflicting partial package (if present)
sudo dnf remove -y golang-github-nvidia-container-toolkit 2>/dev/null || true

# 3. Install the full NVIDIA Container Toolkit
sudo dnf install -y nvidia-container-toolkit

# 4. Configure Docker to use the NVIDIA runtime as default
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "runtimes": {
    "nvidia": {
      "path": "/usr/bin/nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
EOF

# 5. Fix Fedora's broken dockerd ExecStart (required!)
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/override.conf >/dev/null <<EOF
[Service]
ExecStart=
ExecStart=/usr/bin/dockerd -H fd:// --containerd=/run/containerd/containerd.sock
EOF

# 6. Reload and restart Docker
sudo systemctl daemon-reload
sudo systemctl restart docker

# 7. Verify it works
docker info --format '{{.DefaultRuntime}}'   # → should print: nvidia
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```


#### Build Container

`docker build -f ./Dockerfile_ubuntu24 -t ubuntu24:latest .`

`docker build -f ./Dockerfile_fedora42 -t fedora42:latest .`


#### Run Docker Container

Note: If running on CPU, just remove `--gpus all` from commands below:

`docker run -v ~/src/gst-python-ml/:/root/gst-python-ml -it --rm --gpus all --name ubuntu24 ubuntu24:latest /bin/bash`

or

`docker run -v ~/src/gst-python-ml/:/root/gst-python-ml -it --rm --gpus all --name fedora42 fedora42:latest /bin/bash`

Now, in the container shell, set up `uv` `venv` as detailed above.

## IMPORTANT NOTES

### Birdseye

To use `pyml_birdseye`, additional pip requirements must be installed from the `plugins/python/birdseye` folder.


## Post Install

Run `gst-inspect-1.0 python` to list pyml elements.

# Building PyPI Package

## Setup
1. Generate token on PyPI and copy to `.pypirc`

```
[pypi]
  username = __token__
  password = $TOKEN
```

2. Install build dependencies

```
pip install setuptools wheel twine
pip install --upgrade build
```
## Build and Upoad

```
python -m build
twine upload dist/*
```

## Using GStreamer Python ML Elements

## Pipelines

Below are some sample pipelines for the various elements in this project.

### Classification

```
GST_DEBUG=4 gst-launch-1.0  filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_classifier model-name=resnet18 device=cuda !  videoconvert !  autovideosink
```


### Object Detection

#### TorchVision

`pyml_objectdetector` supports all TorchVision  object detection models.
Simply choose a suitable model name and set it on the `model-name` property.
A few possible model names:

```
fasterrcnn_resnet50_fpn
ssdlite320_mobilenet_v3_large
```

##### fasterrcnn

`GST_DEBUG=4 gst-launch-1.0  filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda batch-size=4 ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink`

##### fasterrcnn/kafka

a) run pipeline from host

```
GST_DEBUG=4 gst-launch-1.0  filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda batch-size=4 ! pyml_kafkasink schema-file=data/pyml_object_detector.json broker=localhost:29092 topic=test-kafkasink-topic
```

b) run pipeline from docker

```
GST_DEBUG=4 gst-launch-1.0  filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! pyml_objectdetector model-name=fasterrcnn_resnet50_fpn device=cuda batch-size=4 ! pyml_kafkasink schema-file=data/pyml_object_detector.json broker=kafka:9092 topic=test-kafkasink-topic
```


#### maskrcnn

```
GST_DEBUG=4 gst-launch-1.0   filesrc location=data/people.mp4 ! decodebin ! videoconvert ! videoscale ! pyml_maskrcnn device=cuda batch-size=4 model-name=maskrcnn_resnet50_fpn ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink
```

#### yolo with tracking

```
GST_DEBUG=4 gst-launch-1.0   filesrc location=data/soccer_tracking.mp4 ! decodebin !  videoconvertscale ! video/x-raw,width=640,height=480 ! pyml_yolo model-name=yolo11m device=cuda:0 track=True ! pyml_overlay  ! videoconvert ! autovideosink
```

```
GST_DEBUG=4 gst-launch-1.0   filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480,format=RGB ! pyml_streammux name=mux   filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480,format=RGB ! mux.   mux. ! pyml_yolo model-name=yolo11m device=cuda:0 track=True ! pyml_streamdemux name=demux   demux. ! queue ! videoconvert ! pyml_overlay ! videoconvert ! autovideosink sync=false   demux. ! queue ! videoconvert ! pyml_overlay ! videoconvert !  autovideosink sync=false

```

```
GST_DEBUG=4 gst-launch-1.0 filesrc location=data/soccer_tracking.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480 ! demo_soccer model-name=yolo11m device=cuda:0 ! pyml_overlay ! videoconvert ! autovideosink
```


### Transcription

#### transcription with initial prompt set

```
GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whispertranscribe device=cuda language=ko initial_prompt = "Air Traffic Control은, radar systems를,  weather conditions에, flight paths를, communication은, unexpected weather conditions가, continuous training을, dedication과, professionalism" ! fakesink
```

#### translation to English

```
GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whispertranscribe device=cuda language=ko translate=yes ! fakesink
```

#### demucs audio separation


```
GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! audioresample ! pyml_demucs device=cuda ! wavenc ! filesink location=separated_vocals.wav
```


#### coquitts

```
GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whispertranscribe device=cuda language=ko translate=yes ! pyml_coquitts device=cuda ! audioconvert ! wavenc ! filesink location=output_audio.wav
```

#### whisperspeechtts

```
GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whispertranscribe device=cuda language=ko translate=yes ! pyml_whisperspeechtts device=cuda ! audioconvert ! wavenc ! filesink location=output_audio.wav
```

#### mariantranslate

```
GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whispertranscribe device=cuda language=ko translate=yes ! pyml_mariantranslate device=cuda src=en target=fr ! fakesink
```

Supported src/target languages:

https://huggingface.co/models?sort=trending&search=Helsinki


#### whisperlive

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/air_traffic_korean_with_english.wav ! decodebin ! audioconvert ! pyml_whisperlive device=cuda language=ko translate=yes llm-model-name="microsoft/phi-2" ! audioconvert ! wavenc ! filesink location=output_audio.wav`

### LLM

1. generate HuggingFace token

2. `huggingface-cli login`
    and pass in token

3. LLM pipeline (in this case, we use phi-2)

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/prompt_for_llm.txt !  pyml_llm device=cuda model-name="microsoft/phi-2" ! fakesink`

### stablediffusion

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/prompt_for_stable_diffusion.txt ! pyml_stablediffusion device=cuda ! pngenc ! filesink location=output_image.png`

#### Caption

#### caption qwen with history

(should also work with "microsoft/Phi-3.5-vision-instruct" model)

```
GST_DEBUG=3 gst-launch-1.0 filesrc location=data/soccer_single_camera.mp4 ! decodebin ! videoconvertscale ! video/x-raw,width=640,height=480 ! tee name=t t. ! queue ! textoverlay name=overlay wait-text=false ! videoconvert ! autovideosink t. ! queue leaky=2 max-size-buffers=1 ! videoconvertscale ! video/x-raw,width=240,height=180 ! pyml_caption_qwen device=cuda:0 prompt="In one sentence, describe what you see?" model-name="Qwen/Qwen2.5-VL-3B-Instruct-AWQ" name=cap cap.src ! fakesink async=0 sync=0 cap.text_src ! queue ! coalescehistory history-length=10 ! pyml_llm model-name="Qwen/Qwen3-0.6B" device=cuda system-prompt="You receive the history of what happened in recent times, summarize it nicely with excitement but NEVER mention the specific times. Focus on the most recent events." ! queue ! overlay.text_sink
```


### Bird's Eye View

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/soccer_single_camera.mp4 ! decodebin ! videoconvert ! pyml_birdseye ! videoconvert ! autovideosink`

`GST_DEBUG=4 gst-launch-1.0 filesrc location=data/soccer_single_camera.mp4 ! decodebin ! videorate ! video/x-raw,framerate=30/1 ! videoconvert ! pyml_birdseye ! videoconvert ! openh264enc ! h264parse ! matroskamux ! filesink location=output.mkv`


### kafkasink

#### Setting up kafka network

`docker network create kafka-network`

and list networks

`docker network ls`

#### docker launch

To launch a docker instance with the kafka network, add ` --network kafka-network  `
to the docker launch command above.

#### Set up kafka and zookeeper

Note: setup below assumes you are running your pipeline in a docker container. 
If running pipeline from host, then the port changes from `9092` to `29092`,
and the broker changes from `kafka` to `localhost`.

```
docker stop kafka zookeeper
docker rm kafka zookeeper
docker run -d --name zookeeper --network kafka-network -e ZOOKEEPER_CLIENT_PORT=2181 confluentinc/cp-zookeeper:latest
docker run -d --name kafka --network kafka-network \
  -e KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181 \
  -e KAFKA_ADVERTISED_LISTENERS=INSIDE://kafka:9092,OUTSIDE://localhost:29092 \
  -e KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=INSIDE:PLAINTEXT,OUTSIDE:PLAINTEXT \
  -e KAFKA_LISTENERS=INSIDE://0.0.0.0:9092,OUTSIDE://0.0.0.0:29092 \
  -e KAFKA_INTER_BROKER_LISTENER_NAME=INSIDE \
  -e KAFKA_BROKER_ID=1 \
  -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
  -p 9092:9092 \
  -p 29092:29092 \
  confluentinc/cp-kafka:latest
```

#### Create test topic
```
docker exec kafka kafka-topics --create --topic test-kafkasink-topic --bootstrap-server kafka:9092 --partitions 1 --replication-factor 1
```

#### list topics

`docker exec -it kafka kafka-topics --list --bootstrap-server kafka:9092`


#### delete topic

`docker exec -it kafka kafka-topics --delete --topic test-topic --bootstrap-server kafka:9092`


#### consume topic

`docker exec -it kafka kafka-console-consumer --bootstrap-server kafka:9092 --topic test-kafkasink-topic --from-beginning`


### non ML

`GST_DEBUG=4 gst-launch-1.0 videotestsrc ! video/x-raw,width=1280,height=720 ! pyml_overlay meta-path=data/sample_metadata.json tracking=true ! videoconvert ! autovideosink`


### streammux/streamdemux pipeline

```
 GST_DEBUG=4 gst-launch-1.0   videotestsrc pattern=ball ! video/x-raw, width=320, height=240 ! queue ! pyml_streammux name=mux   videotestsrc pattern=smpte ! video/x-raw, width=320, height=240 ! queue ! mux.sink_1   videotestsrc pattern=smpte ! video/x-raw, width=320, height=240 ! queue ! mux.sink_2   mux.src ! queue ! pyml_streamdemux name=demux   demux.src_0 ! queue ! glimagesink  demux.src_1 ! queue ! glimagesink   demux.src_2 ! queue  ! glimagesink
```