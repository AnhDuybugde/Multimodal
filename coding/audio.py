from __future__ import annotations

import math
import wave

import torch
import torch.nn.functional as F


class AudioPipeline:
    def __init__(
        self,
        target_sample_rate: int = 22000,
        window_sec: float = 0.8,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 256,
        normalize_db: bool = True,
    ) -> None:
        try:
            import torchaudio
        except (ImportError, OSError):
            torchaudio = None

        self.torchaudio = torchaudio
        self.target_sample_rate = target_sample_rate
        self.window_samples = int(round(target_sample_rate * window_sec))
        self.normalize_db = normalize_db
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        if torchaudio is not None:
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=target_sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0,
            )
            self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
            self.mel_filter = None
        else:
            self.mel = None
            self.to_db = None
            self.mel_filter = _mel_filterbank(target_sample_rate, n_fft, n_mels)
        self._resamplers: dict[int, torch.nn.Module] = {}

    def load_waveform(self, path: str):
        if self.torchaudio is not None:
            waveform, sample_rate = self.torchaudio.load(path)
        else:
            waveform, sample_rate = _load_wav_stdlib(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.target_sample_rate:
            if self.torchaudio is not None and sample_rate not in self._resamplers:
                self._resamplers[sample_rate] = self.torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.target_sample_rate,
                    lowpass_filter_width=64,
                    rolloff=0.9475937167399596,
                    resampling_method="sinc_interp_kaiser",
                )
            if self.torchaudio is not None:
                waveform = self._resamplers[sample_rate](waveform)
            else:
                waveform = _resample_scipy(waveform, sample_rate, self.target_sample_rate)
        return waveform

    def crop_or_pad(self, waveform: torch.Tensor, mode: str = "center") -> torch.Tensor:
        total = waveform.shape[-1]
        target = self.window_samples
        if total < target:
            return F.pad(waveform, (0, target - total))
        if total == target:
            return waveform

        max_start = total - target
        if mode == "random":
            start = int(torch.randint(0, max_start + 1, (1,)).item())
        elif mode == "energy":
            start = self._energy_start(waveform, target)
        else:
            start = max_start // 2
        return waveform[..., start : start + target]

    def _energy_start(self, waveform: torch.Tensor, target: int) -> int:
        # For 1s -> 0.8s crops this is cheap and picks the loudest contiguous window.
        energy = waveform.pow(2).mean(dim=0, keepdim=True).unsqueeze(0)
        kernel = torch.ones(1, 1, target, device=waveform.device)
        scores = F.conv1d(energy, kernel)
        return int(scores.argmax(dim=-1).item())

    def waveform_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.torchaudio is not None:
            mel = self.mel(waveform)
            mel = self.to_db(mel)
        else:
            window = torch.hann_window(self.n_fft, device=waveform.device)
            spec = torch.stft(
                waveform,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=window,
                return_complex=True,
            ).abs().pow(2.0)
            mel_filter = self.mel_filter.to(waveform.device)
            mel = torch.matmul(mel_filter, spec.squeeze(0)).unsqueeze(0)
            mel = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
            mel = torch.clamp(mel, min=mel.max() - 80.0)
        if self.normalize_db:
            mel = (mel + 80.0) / 80.0
            mel = mel.clamp(0.0, 1.0)
        return mel

    def __call__(self, path: str, crop_mode: str = "center") -> torch.Tensor:
        waveform = self.load_waveform(path)
        waveform = self.crop_or_pad(waveform, crop_mode)
        return self.waveform_to_mel(waveform)


def _load_wav_stdlib(path: str):
    with wave.open(path, "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 2:
        dtype = torch.int16
        scale = float(2**15)
    elif sample_width == 1:
        dtype = torch.uint8
        scale = 255.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    data = torch.frombuffer(bytearray(frames), dtype=dtype).float()
    if sample_width == 1:
        data = data - 128.0
    data = data.view(-1, channels).t().contiguous() / scale
    return data, sample_rate


def _resample_scipy(waveform: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise ImportError("Install torchaudio or scipy for audio resampling.") from exc

    gcd = math.gcd(orig_freq, new_freq)
    up = new_freq // gcd
    down = orig_freq // gcd
    resampled = resample_poly(waveform.numpy(), up=up, down=down, axis=-1)
    return torch.from_numpy(resampled.copy()).float()


def _hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq / 700.0)


def _mel_to_hz(mels: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    n_freqs = n_fft // 2 + 1
    min_mel = _hz_to_mel(torch.tensor(0.0))
    max_mel = _hz_to_mel(torch.tensor(float(sample_rate) / 2.0))
    mels = torch.linspace(min_mel, max_mel, n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = torch.floor((n_fft + 1) * hz / sample_rate).long().clamp(0, n_freqs - 1)

    fb = torch.zeros(n_mels, n_freqs)
    for i in range(n_mels):
        left, center, right = bins[i].item(), bins[i + 1].item(), bins[i + 2].item()
        if center > left:
            fb[i, left:center] = torch.linspace(0.0, 1.0, center - left)
        if right > center:
            fb[i, center:right] = torch.linspace(1.0, 0.0, right - center)
    return fb
