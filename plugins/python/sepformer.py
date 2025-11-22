# Sepformer (assuming this is sepformer.py based on traceback)
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
    from gi.repository import Gst, GObject, GstBase  # noqa: E402
    from base_separate import BaseSeparate
    import torch
    from speechbrain.pretrained import SepformerSeparation
    from torchaudio.transforms import Fade, Resample
    from huggingface_hub import snapshot_download
    import os
    import numpy as np

    from engine.pytorch_engine import PyTorchEngine
    from engine.engine_factory import EngineFactory

except ImportError as e:
    CAN_REGISTER_ELEMENT = False
    GlobalLogger().warning(
        f"The 'pyml_sepformer' element will not be available. Error: {e}"
    )


class SepformerEngine(PyTorchEngine):
    def __init__(self):
        super().__init__()
        self.sample_rate = 0

    def do_load_model(self, model_name, **kwargs):
        if not model_name:
            return
        self.logger.info(f"Loading Sepformer-WhamR model on device: {self.device}")
        savedir = "pretrained_models/sepformer-whamr"
        repo_id = "speechbrain/sepformer-whamr"
        try:
            # Download the model files manually to avoid deprecated argument issues
            if not os.path.exists(savedir):
                snapshot_download(repo_id=repo_id, local_dir=savedir)
            # Load from local directory
            self.model = SepformerSeparation.from_hparams(
                source=savedir, savedir=savedir, run_opts={"device": self.device}
            )
            self.sample_rate = 8000  # Hz, as per SpeechBrain Sepformer models
            self.sources = ["source0", "source1"]  # 2 sources for separation
        except Exception as e:
            self.logger.error(f"Failed to load Sepformer-WhamR model: {e}")

    def separate_sources(
        self,
        mix,
        segment=10.0,
        overlap=0.1,
    ):
        device = mix.device
        batch, length = mix.shape  # For SpeechBrain, input is (batch, time)
        chunk_len = int(self.sample_rate * segment * (1 + overlap))
        start = 0
        end = chunk_len
        overlap_frames = int(overlap * self.sample_rate)
        fade = Fade(fade_in_len=0, fade_out_len=overlap_frames, fade_shape="linear")

        final = torch.zeros(batch, len(self.sources), length, device=device)

        min_chunk_samples = int(self.sample_rate * 0.5)  # Avoid tiny chunks

        while start < length - overlap_frames:
            actual_end = min(end, length)
            chunk_length = actual_end - start
            if chunk_length < min_chunk_samples:
                break

            chunk = mix[:, start:actual_end]
            if chunk_length < chunk_len:
                pad = chunk_len - chunk_length
                chunk = torch.nn.functional.pad(chunk, (0, pad))

            # Add small epsilon noise to avoid zero std
            chunk += 1e-8 * torch.randn_like(chunk)

            with torch.no_grad():
                out = self.model.separate_batch(chunk)  # (batch, time, sources)
                out = out.permute(0, 2, 1)  # (batch, sources, time)

            out = out[:, :, :chunk_length]
            out = fade(out)
            final[:, :, start:actual_end] += out
            if start == 0:
                fade.fade_in_len = overlap_frames
                start += int(chunk_len - overlap_frames)
            else:
                start += chunk_len
            end += chunk_len
            if end >= length:
                fade.fade_out_len = 0
        return final


class Sepformer(BaseSeparate):
    __gstmetadata__ = (
        "Sepformer",
        "Audio Output",
        "Python element that separates speakers with RE-SepFormer in noisy environments",
        "Aaron Boxer <aaron.boxer@collabora.com>",
    )

    SAMPLE_RATE = 8000

    CAPS = Gst.Caps(
        Gst.Structure(
            "audio/x-raw",
            format="S16LE",
            layout="interleaved",
            rate=SAMPLE_RATE,
            channels=1,
        )
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
        self.model_name = "pyml_sepformer_model"
        self.mgr.engine_name = "pyml_sepformer_engine"
        EngineFactory.register(self.mgr.engine_name, SepformerEngine)

    @GObject.Property(type=str)
    def engine_name(self):
        return self.mgr.engine_name

    @engine_name.setter
    def engine_name(self, value):
        raise ValueError("engine_name cannot be set")

    def do_separate(self, audio_data):
        engine = self.engine
        if engine.model is None:
            engine.do_load_model(self.model_name)

        audio_torch = torch.from_numpy(audio_data).float().to(engine.device)
        mixture = audio_torch.unsqueeze(0)  # (1, length) for SpeechBrain

        try:
            sources = engine.separate_sources(
                mixture,
                segment=(
                    15.0 if not self.streaming else 1.0
                ),  # Increased for better quality
                overlap=0.1,
            )  # (batch, sources, length)
        except Exception as e:
            self.logger.error(f"Separation failed: {e}")
            return np.zeros(len(audio_data), dtype=np.float32)

        energies = torch.mean(sources**2, dim=[1, 2])
        if energies.sum() == 0:
            idx = 0  # Default to first source if all energies zero
        else:
            idx = torch.argmax(energies)
        selected = sources[0, idx, :]  # batch=0

        # Normalize to unit amplitude for better volume
        max_amp = selected.abs().max()
        if max_amp > 0:
            selected = selected / max_amp

        return selected.cpu().numpy()


if CAN_REGISTER_ELEMENT:
    GObject.type_register(Sepformer)
    __gstelementfactory__ = ("pyml_sepformer", Gst.Rank.NONE, Sepformer)
else:
    GlobalLogger().warning("pyml_sepformer not registered")
