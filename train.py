from __future__ import annotations

import argparse
import json
import random
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from kokoro import KModel
from kokoro.th import THG2P


MAX_PACK_LEN = 510
DEFAULT_INIT_VOICE = "af_heart"


@dataclass
class Sample:
    stem: str
    wav_path: Path
    text_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a full Kokoro voice pack for Thai while freezing the full model."
    )
    parser.add_argument("--repo-id", default="hexgrad/Kokoro-82M")
    parser.add_argument("--wav-dir", type=Path, default=Path("dataset/wav"))
    parser.add_argument("--text-dir", type=Path, default=Path("dataset/wrd_ph"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/voices/custom_pack.pt"))
    parser.add_argument("--init-voice", default=DEFAULT_INIT_VOICE)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--wave-loss-weight", type=float, default=1.0)
    parser.add_argument("--stft-loss-weight", type=float, default=1.0)
    parser.add_argument("--length-loss-weight", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="cuda, cpu, or mps. Defaults to auto.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle sample order each epoch.",
    )
    return parser.parse_args()


def choose_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pair_samples(wav_dir: Path, text_dir: Path, limit: Optional[int]) -> list[Sample]:
    wav_files = {p.stem: p for p in sorted(wav_dir.glob("*.wav"))}
    text_files = {p.stem: p for p in sorted(text_dir.glob("*.txt"))}
    shared = sorted(set(wav_files) & set(text_files))
    if limit is not None:
        shared = shared[:limit]
    if not shared:
        raise FileNotFoundError(f"No paired .wav/.txt files found under {wav_dir} and {text_dir}")
    return [Sample(stem=stem, wav_path=wav_files[stem], text_path=text_files[stem]) for stem in shared]


def read_text_line(path: Path, line_number: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not 1 <= line_number <= len(lines):
        raise ValueError(f"{path} has {len(lines)} lines, cannot read line {line_number}")
    return lines[line_number - 1].strip()


def strip_pipes(text: str) -> str:
    return text.replace("|", "")


def clean_text_files(text_dir: Path) -> int:
    rewritten = 0
    for path in sorted(text_dir.glob("*.txt")):
        content = path.read_text(encoding="utf-8")
        cleaned = strip_pipes(content)
        if cleaned != content:
            path.write_text(cleaned, encoding="utf-8")
            rewritten += 1
    return rewritten


def load_wav_mono(path: Path, target_sample_rate: int) -> torch.Tensor:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
    if sample_width not in dtype_map:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes in {path}")

    audio = np.frombuffer(raw, dtype=dtype_map[sample_width])
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if sample_width == 1:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32) / 2147483648.0

    waveform = torch.from_numpy(audio.copy())
    if sample_rate != target_sample_rate:
        waveform = resample_1d(waveform, sample_rate, target_sample_rate)
    return waveform.clamp(-1.0, 1.0)


def resample_1d(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    if source_rate == target_rate:
        return waveform
    target_len = max(1, round(waveform.numel() * target_rate / source_rate))
    resampled = F.interpolate(
        waveform[None, None, :],
        size=target_len,
        mode="linear",
        align_corners=False,
    )
    return resampled[0, 0]


def load_initial_pack(repo_id: str, init_voice: str) -> torch.Tensor:
    voice_path = Path(init_voice)
    if voice_path.exists():
        loaded = torch.load(voice_path, map_location="cpu", weights_only=True).float()
    else:
        filename = f"voices/{init_voice}.pt"
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename)
        loaded = torch.load(downloaded, map_location="cpu", weights_only=True).float()

    if tuple(loaded.shape) == (MAX_PACK_LEN, 1, 256):
        return loaded
    if tuple(loaded.shape) == (1, 256):
        return loaded.unsqueeze(0).repeat(MAX_PACK_LEN, 1, 1)
    if tuple(loaded.shape) == (256,):
        return loaded.unsqueeze(0).unsqueeze(0).repeat(MAX_PACK_LEN, 1, 1)
    raise ValueError(f"Unsupported initial pack shape: {tuple(loaded.shape)}")


def freeze_model(model: KModel) -> None:
    # Keep the model in training mode so cuDNN LSTM backward works while
    # gradients still flow into the trainable pack.
    model.train()
    for param in model.parameters():
        param.requires_grad_(False)
    # Disable stochastic dropout because we are not training model weights.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()


def phonemes_to_input_ids(model: KModel, phonemes: str, device: str) -> tuple[str, torch.Tensor]:
    phonemes = phonemes.strip()
    if len(phonemes) > MAX_PACK_LEN:
        phonemes = phonemes[:MAX_PACK_LEN]

    input_ids = [model.vocab[p] for p in phonemes if p in model.vocab]
    if not input_ids:
        raise ValueError("No model vocab symbols remained after phoneme filtering")

    tensor = torch.tensor([[0, *input_ids, 0]], dtype=torch.long, device=device)
    return phonemes, tensor


def forward_with_trainable_pack(
    model: KModel,
    input_ids: torch.Tensor,
    ref_s: torch.Tensor,
    speed: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_lengths = torch.full(
        (input_ids.shape[0],),
        input_ids.shape[-1],
        device=input_ids.device,
        dtype=torch.long,
    )

    text_mask = torch.arange(input_lengths.max(), device=input_ids.device).unsqueeze(0)
    text_mask = text_mask.expand(input_lengths.shape[0], -1)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1)).to(model.device)

    bert_dur = model.bert(input_ids, attention_mask=(~text_mask).int())
    d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

    s = ref_s[:, 128:]
    d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
    x, _ = model.predictor.lstm(d)
    duration = model.predictor.duration_proj(x)
    duration = torch.sigmoid(duration).sum(axis=-1) / speed
    pred_dur = torch.round(duration).clamp(min=1).long().squeeze()

    indices = torch.repeat_interleave(
        torch.arange(input_ids.shape[1], device=model.device),
        pred_dur,
    )
    pred_aln_trg = torch.zeros((input_ids.shape[1], indices.shape[0]), device=model.device)
    pred_aln_trg[indices, torch.arange(indices.shape[0], device=model.device)] = 1
    pred_aln_trg = pred_aln_trg.unsqueeze(0)

    en = d.transpose(-1, -2) @ pred_aln_trg
    f0_pred, n_pred = model.predictor.F0Ntrain(en, s)
    t_en = model.text_encoder(input_ids, input_lengths, text_mask)
    asr = t_en @ pred_aln_trg
    audio = model.decoder(asr, f0_pred, n_pred, ref_s[:, :128]).squeeze()
    return audio, pred_dur


def stft_magnitude(signal: torch.Tensor, n_fft: int) -> Optional[torch.Tensor]:
    if signal.numel() < n_fft:
        return None
    window = torch.hann_window(n_fft, device=signal.device)
    spec = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=n_fft // 4,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )
    return spec.abs()


def pack_loss(
    pred_audio: torch.Tensor,
    target_audio: torch.Tensor,
    sample_rate: int,
    wave_weight: float,
    stft_weight: float,
    length_weight: float,
) -> torch.Tensor:
    pred_audio = pred_audio.float()
    target_audio = target_audio.float()

    min_len = min(pred_audio.numel(), target_audio.numel())
    if min_len <= 0:
        raise ValueError("Encountered empty audio while computing loss")

    pred_crop = pred_audio[:min_len]
    target_crop = target_audio[:min_len]

    wave_loss = F.l1_loss(pred_crop, target_crop)

    stft_loss = pred_crop.new_tensor(0.0)
    for n_fft in (512, 1024, 2048):
        pred_mag = stft_magnitude(pred_crop, n_fft)
        target_mag = stft_magnitude(target_crop, n_fft)
        if pred_mag is None or target_mag is None:
            continue
        stft_loss = stft_loss + F.l1_loss(pred_mag, target_mag)

    length_loss = pred_crop.new_tensor(abs(pred_audio.numel() - target_audio.numel()) / sample_rate)
    return wave_weight * wave_loss + stft_weight * stft_loss + length_weight * length_loss


def extract_phonemes(
    sample: Sample,
    thai_g2p: THG2P,
) -> str:
    text = strip_pipes(read_text_line(sample.text_path, 1)).strip()
    if not text:
        raise ValueError("text line 1 is empty after removing '|' characters")
    phonemes, _ = thai_g2p(text)
    return phonemes.strip()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    samples = pair_samples(args.wav_dir, args.text_dir, args.limit)
    rewritten_files = clean_text_files(args.text_dir)
    thai_g2p = THG2P()

    model = KModel(repo_id=args.repo_id).to(device)
    freeze_model(model)

    init_pack = load_initial_pack(args.repo_id, args.init_voice)
    if tuple(init_pack.shape) != (MAX_PACK_LEN, 1, 256):
        raise ValueError(f"Expected initial pack with shape ({MAX_PACK_LEN}, 1, 256), got {tuple(init_pack.shape)}")

    trainable_pack = torch.nn.Parameter(init_pack.to(device))
    optimizer = torch.optim.AdamW([trainable_pack], lr=args.lr, weight_decay=args.weight_decay)

    print(f"paired samples: {len(samples)}")
    print(f"device: {device}")
    print(f"cleaned text files: {rewritten_files}")
    print(f"trainable pack shape: {tuple(trainable_pack.shape)}")

    best_loss = float("inf")
    best_pack = trainable_pack.detach().cpu().clone()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        seen = 0
        skipped = 0
        updates = 0

        epoch_samples = list(samples)
        if args.shuffle:
            random.shuffle(epoch_samples)

        for batch_start in range(0, len(epoch_samples), args.batch_size):
            batch_samples = epoch_samples[batch_start:batch_start + args.batch_size]
            batch_losses = []
            batch_seen = 0

            optimizer.zero_grad(set_to_none=True)

            for sample in batch_samples:
                try:
                    phonemes = extract_phonemes(sample, thai_g2p)
                    phonemes, input_ids = phonemes_to_input_ids(model, phonemes, device)
                    ref_s = trainable_pack[len(phonemes) - 1]
                    target_audio = load_wav_mono(sample.wav_path, args.sample_rate).to(device)
                except Exception as exc:
                    skipped += 1
                    print(f"[skip] {sample.stem}: {exc}")
                    continue

                pred_audio, _ = forward_with_trainable_pack(model, input_ids, ref_s, speed=1.0)
                loss = pack_loss(
                    pred_audio=pred_audio,
                    target_audio=target_audio,
                    sample_rate=args.sample_rate,
                    wave_weight=args.wave_loss_weight,
                    stft_weight=args.stft_loss_weight,
                    length_weight=args.length_loss_weight,
                )
                batch_losses.append(loss)
                epoch_loss += float(loss.detach().cpu())
                seen += 1
                batch_seen += 1

            if not batch_losses:
                continue

            batch_loss = torch.stack(batch_losses).mean()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_([trainable_pack], args.grad_clip)
            optimizer.step()
            updates += 1

            if updates % args.log_every == 0:
                print(
                    f"epoch {epoch}/{args.epochs} update {updates} "
                    f"batch_start={batch_start + 1}/{len(epoch_samples)} "
                    f"batch_size={batch_seen} batch_loss={float(batch_loss.detach().cpu()):.6f}"
                )

        if seen == 0:
            raise RuntimeError("No training samples were usable. Check phoneme extraction and dataset paths.")

        avg_loss = epoch_loss / seen
        print(
            f"epoch {epoch} complete: avg_loss={avg_loss:.6f} "
            f"seen={seen} skipped={skipped} updates={updates}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_pack = trainable_pack.detach().cpu().clone()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_pack, args.output)
    meta_path = args.output.with_suffix(args.output.suffix + ".json")
    meta_path.write_text(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "init_voice": args.init_voice,
                "g2p": "kokoro.th.THG2P",
                "text_line": 1,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "sample_rate": args.sample_rate,
                "best_loss": best_loss,
                "samples": len(samples),
                "cleaned_text_files": rewritten_files,
                "pack_shape": list(best_pack.shape),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"saved trained pack: {args.output}")
    print(f"saved metadata: {meta_path}")


if __name__ == "__main__":
    main()
