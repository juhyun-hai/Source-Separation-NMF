import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf

import mir_eval

def plot_time_domain_signals(
    mixture_signal: np.ndarray,
    reference_signal: np.ndarray,
    estimated_signal: np.ndarray
):
    
    plt.figure(figsize=(12, 6))
    # 1) Bearing vs Voice
    plt.subplot(3, 1, 1)
    plt.plot(reference_signal[0,:], label="Bearing")
    plt.plot(reference_signal[1,:], label="Voice", alpha=0.7)
    plt.title("Bearing, Voice Signals")
    plt.legend(loc="upper right")

    # 2) Mixed
    plt.subplot(3, 1, 2)
    plt.plot(mixture_signal, label="Mixture")
    plt.title("Mixed Signal")
    plt.legend(loc="upper right")

    # 3) Separated
    plt.subplot(3, 1, 3)
    plt.plot(estimated_signal[0, :], label="Estimated Bearing")
    plt.plot(estimated_signal[1, :], label="Estimated Voice", alpha=0.7)
    plt.title("Separated Sources (NMF)")
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()

def calc_bss_metrics(reference: np.ndarray, estimated: np.ndarray):
    sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(reference, estimated)

    # (추가) mixture를 기준으로 SDRi 계산
    # 믹스가 두 소스가 같은 신호라고 가정할 경우 (stack)
    mix_sdr, _, _, _ = mir_eval.separation.bss_eval_sources(
        reference, np.stack([reference[0] + reference[1], reference[0] + reference[1]])
    )
    sdri = sdr - mix_sdr

    metrics = {
        "SDR": sdr,
        "SIR": sir,
        "SAR": sar,
        "SDRi": sdri,
    }
    return metrics

def plot_logmel_spectrogram(signal, title="Log-Mel", sr=44100, hop_length=512, n_fft=1024, n_mels=128):
    """
    Log-mel 스펙트로그램을 시각화하는 함수.
    """
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(logmel, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"[Log-Mel] {title}")
    plt.tight_layout()
    plt.show()