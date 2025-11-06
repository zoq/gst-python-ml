# BaseSeparate
# Copyright (C) 2024-2025 Collabora Ltd.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Library General Public License for more details.
#
# You should have received a copy of the GNU Library General Public
# License along with this library; if not, write to the
# Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
# Boston, MA 02110-1301, USA.

import collections
import sys
from abc import abstractmethod

import numpy as np
from pysilero_vad import SileroVoiceActivityDetector
import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstBase", "1.0")
gi.require_version("GObject", "2.0")
from gi.repository import Gst, GObject, GstBase  # noqa: E402

from base_aggregator import BaseAggregator  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

SAMPLE_RATE = 16000  # Target sample rate for processing

# Initialize VAD
vad = SileroVoiceActivityDetector()
vad_chunk_size = vad.chunk_samples()

CAPS = Gst.Caps(
    Gst.Structure(
        "audio/x-raw",
        format="S16LE",
        layout="interleaved",
        rate=SAMPLE_RATE,
        channels=1,
    )
)


class BaseSeparate(BaseAggregator):
    __gstmetadata__ = (
        "BaseSeparate",
        "Audio Output",
        "Python element that separates audio sources",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    __gsttemplates__ = (
        Gst.PadTemplate.new_with_gtype(
            "sink",
            Gst.PadDirection.SINK,
            Gst.PadPresence.REQUEST,
            CAPS,
            GstBase.AggregatorPad.__gtype__,
        ),
        Gst.PadTemplate.new_with_gtype(
            "src",
            Gst.PadDirection.SRC,
            Gst.PadPresence.ALWAYS,
            CAPS,
            GstBase.AggregatorPad.__gtype__,
        ),
    )

    def __init__(self):
        super().__init__()

        self.clip_buffer = collections.deque()
        self.active_clip = False
        self.silence_counter = 0
        chunk_duration_ms = (vad_chunk_size / SAMPLE_RATE) * 1000
        silence_ms = 300
        self.clip_silence_trigger_counter = int(silence_ms / chunk_duration_ms)
        self.__streaming = False
        self.__stem = "vocals"

    @GObject.Property(type=bool, default=False)
    def streaming(self):
        "toggle streaming"
        return self.__streaming

    @streaming.setter
    def streaming(self, value):
        self.__streaming = value

    @GObject.Property(type=str, default="vocals")
    def stem(self):
        "stem to output (e.g., vocals, drums, bass, other)"
        return self.__stem

    @stem.setter
    def stem(self, value):
        self.__stem = value

    @abstractmethod
    def do_load_model(self):
        pass

    @abstractmethod
    def do_separate(self, audio_data):
        pass

    def do_process_audio(self, audio_data):
        # audio_data is np.int16 array
        encoded_size = audio_data.nbytes
        outbuf = Gst.Buffer.new_allocate(None, encoded_size, None)
        outbuf.fill(0, audio_data.tobytes())
        return outbuf

    def do_process(self, buf):
        self.push_segment_if_needed()
        """Process audio data from the input buffers using VAD and source separation."""
        audio_collected = False

        try:
            self.do_load_model()

            # Map the buffer to access the audio data
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                self.logger.error("Failed to map input buffer")
                buf.unmap(map_info)
                return Gst.FlowReturn.OK

            # Convert buffer to numpy array (int16)
            audio_data = np.frombuffer(map_info.data, dtype=np.int16)
            audio_collected = True

            if len(audio_data) < vad_chunk_size:
                self.logger.warning("Insufficient audio data for processing")
                buf.unmap(map_info)
                return Gst.FlowReturn.OK

            # Process audio data with VAD (Voice Activity Detection)
            while len(audio_data) >= vad_chunk_size:
                vad_chunk = audio_data[:vad_chunk_size]
                audio_data = audio_data[vad_chunk_size:]

                vad_confidence = vad.process_chunk(vad_chunk.tobytes())
                if vad_confidence >= 0.7:
                    if self.streaming:
                        separated = self._separate_audio(vad_chunk)
                        if separated is None:
                            self.logger.warning("Empty separated audio")
                            buf.unmap(map_info)
                            return Gst.FlowReturn.ERROR

                        self._process_and_send(separated, buf)
                    else:
                        # VAD detects voice activity, add to buffer
                        self.active_clip = True
                        self.silence_counter = 0
                        self.clip_buffer.extend(vad_chunk)
                else:
                    # Increment silence counter when no voice is detected
                    self.silence_counter += 1

                    # If silence is detected for too long, end the current segment
                    if (
                        self.active_clip
                        and self.silence_counter > self.clip_silence_trigger_counter
                    ):
                        self.active_clip = False
                        if not self.streaming:
                            # Perform separation in batch mode
                            separated = self._separate_audio(self.clip_buffer)
                            if separated is None:
                                self.logger.warning("Empty separated audio")
                                buf.unmap(map_info)
                                return Gst.FlowReturn.ERROR

                            self._process_and_send(separated, buf)
                            self.clip_buffer.clear()  # Clear the buffer for the next speech
            buf.unmap(map_info)
        except Exception as e:
            self.logger.error(f"Error during buffer processing: {e}")

        if not audio_collected:
            self.logger.warning("No audio data collected from sink pads.")
            return Gst.FlowReturn.ERROR

        return Gst.FlowReturn.OK

    def _process_and_send(self, separated, inbuf):
        outbuf = self.do_process_audio(separated)
        if outbuf is None:
            return

        # Set PTS and duration from the input buffer
        outbuf.pts = inbuf.pts
        outbuf.duration = inbuf.duration

        # Push the separated audio downstream
        self.finish_buffer(outbuf)

    def _separate_audio(self, chunk):
        """
        Separates the buffered audio data
        and returns the separated stem as np.int16.
        """
        try:
            # Get the current audio data from the buffer for separation
            audio_data = np.array(chunk).astype(np.float32) / 32768.0
            result = self.do_separate(audio_data)
            # Convert back to int16
            separated = np.clip(result * 32768, -32768, 32767).astype(np.int16)
            self.logger.info(f"Separated audio length: {len(separated)}")
            return separated

        except Exception as e:
            self.logger.error(f"Error during separation: {e}")
            return None
