# Demucs
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

from log.global_logger import GlobalLogger

CAN_REGISTER_ELEMENT = True
try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstBase", "1.0")
    gi.require_version("GObject", "2.0")
    from gi.repository import Gst, GObject  # noqa: E402
    from base_separate import BaseSeparate
    import torch
    from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
    from torchaudio.transforms import Fade, Resample

    from engine.pytorch_engine import PyTorchEngine
    from engine.engine_factory import EngineFactory

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_demucs' element will not be available. Error: {e}"
    )


class DemucsEngine(PyTorchEngine):
    def __init__(self):
        super().__init__()
        self.sample_rate = 0

    def do_load_model(self, model_name, **kwargs):
        if not model_name:
            return
        self.logger.info(f"Loading Demucs model on device: {self.device}")
        bundle = (
            HDEMUCS_HIGH_MUSDB_PLUS  # You can choose other bundles like DEMUCS_MUSDB
        )
        self.model = bundle.get_model()
        if hasattr(self.model, "to") and callable(getattr(self.model, "to")):
            self.model = self.model.to(self.device)
        self.sample_rate = bundle.sample_rate  # 44100 Hz
        self.sources = self.model.sources  # ['drums', 'bass', 'other', 'vocals']

    def separate_sources(
        self,
        mix,
        segment=10.0,
        overlap=0.1,
    ):
        device = mix.device
        batch, channels, length = mix.shape
        chunk_len = int(self.sample_rate * segment * (1 + overlap))
        start = 0
        end = chunk_len
        overlap_frames = int(overlap * self.sample_rate)
        fade = Fade(fade_in_len=0, fade_out_len=overlap_frames, fade_shape="linear")

        final = torch.zeros(batch, len(self.sources), channels, length, device=device)

        while start < length - overlap_frames:
            chunk = mix[:, :, start:end]
            with torch.no_grad():
                out = self.model.forward(chunk)
            out = fade(out)
            final[:, :, :, start:end] += out
            if start == 0:
                fade.fade_in_len = overlap_frames
                start += int(chunk_len - overlap_frames)
            else:
                start += chunk_len
            end += chunk_len
            if end >= length:
                fade.fade_out_len = 0
        return final


class Demucs(BaseSeparate):
    __gstmetadata__ = (
        "Demucs",
        "Audio Output",
        "Python element that separates audio sources with Demucs",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    def __init__(self):
        super().__init__()
        self.model_name = "pyml_demucs_model"  # Placeholder, since we use bundle
        self.mgr.engine_name = "pyml_demucs_engine"
        EngineFactory.register(self.mgr.engine_name, DemucsEngine)

    # make engine_name read only
    @GObject.Property(type=str)
    def engine_name(self):
        """Machine Learning Engine (read-only in this class)."""
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError(
            "The 'engine_name' property cannot be set in this derived class."
        )

    def do_separate(self, audio_data):
        # audio_data: np.float32, shape (length,) at SAMPLE_RATE Hz mono
        engine = self.engine
        original_rate = self.SAMPLE_RATE

        # Resample to model's sample rate (44100 Hz) if necessary
        if original_rate != engine.sample_rate:
            resample = Resample(original_rate, engine.sample_rate).to(engine.device)
            audio_torch = torch.from_numpy(audio_data).float().to(engine.device)
            audio_resampled = resample(audio_torch)
        else:
            audio_torch = torch.from_numpy(audio_data).float().to(engine.device)
            audio_resampled = audio_torch

        # Convert mono to stereo by duplicating channel
        audio_stereo = audio_resampled.repeat(2, 1)  # (2, length)

        # Add batch dimension: (1, 2, length)
        mixture = audio_stereo.unsqueeze(0)

        # Separate sources with chunking
        sources = engine.separate_sources(
            mixture,
            segment=(
                10.0 if not self.streaming else 1.0
            ),  # Smaller segments for streaming
            overlap=0.1,
        )[
            0
        ]  # Remove batch dim: (num_sources, 2, length)

        # Select the desired stem
        idx = engine.sources.index(self.stem)
        selected = sources[idx]  # (2, length)

        # Average to mono
        selected_mono = selected.mean(0)  # (length,)

        # Resample back to original rate if necessary
        if original_rate != engine.sample_rate:
            resample_back = Resample(engine.sample_rate, original_rate).to(
                engine.device
            )
            selected_resampled = resample_back(selected_mono)
        else:
            selected_resampled = selected_mono

        return selected_resampled.cpu().numpy()  # float32


if CAN_REGISTER_ELEMENT:
    GObject.type_register(Demucs)
    __gstelementfactory__ = ("pyml_demucs", Gst.Rank.NONE, Demucs)
else:
    GlobalLogger().warning(
        "The 'pyml_demucs' element will not be registered because base_separate module is missing."
    )
