# inference.py
import torch
import torchaudio
import torch.nn.functional as F

from model import LCNN

# Same LFCC settings used in training
def extract_lfcc(wav_path, sample_rate=16000, n_lfcc=60):
    waveform, sr = torchaudio.load(wav_path)

    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    wav_1d = waveform.squeeze(0)

    lfcc_transform = torchaudio.transforms.LFCC(
        sample_rate=sample_rate,
        n_lfcc=n_lfcc,
        speckwargs={
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "center": True,
            "power": 2.0,
        },
    )

    lfcc = lfcc_transform(wav_1d)  # (F, T)

    # Normalize (same as dataset)
    mean = lfcc.mean(dim=1, keepdim=True)
    std = lfcc.std(dim=1, keepdim=True) + 1e-6
    lfcc = (lfcc - mean) / std

    lfcc = lfcc.unsqueeze(0).unsqueeze(0)  # (1, 1, F, T)

    return lfcc


def predict(wav_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = LCNN(in_channels=1, num_classes=2)
    model.load_state_dict(torch.load("best_lfcc_lcnn.pt", map_location=device))
    model.to(device)
    model.eval()

    # Extract LFCC
    lfcc = extract_lfcc(wav_path).to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(lfcc)
        probs = F.softmax(logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item()

    label_map = {0: "Bona Fide (Real)", 1: "Spoof (Fake)"}

    print(f"\nFile: {wav_path}")
    print(f"Prediction: {label_map[pred]}")
    print(f"Confidence: {confidence:.4f}")

    return pred, confidence


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", type=str, help="Path to .wav file")
    args = parser.parse_args()

    predict(args.audio_path)