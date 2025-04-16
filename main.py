# main.py 예시
import numpy as np
from argument_parser import get_args
from src.data_preprocessing import load_and_prepare_audio, train_test
from src.nmf_model import train_nmf_dictionary, separate_signals_with_nmf
from src.utils import (plot_time_domain_signals, calc_bss_metrics, plot_logmel_spectrogram)

def main():

    args = get_args()

    audio_dict = load_and_prepare_audio(args.voice_path, args.bearing_normal_path, args.bearing_fault_path, trim=True)

    train_voice, test_voice, train_bearing, test_bearing, train_mixture, test_mixture = train_test(audio_dict)

    Basis1 = train_nmf_dictionary(train_bearing, sample_rate=args.sample_rate, n_fft=args.n_fft, hop_length=args.hop_length, n_components=args.n_components)
    Basis2 = train_nmf_dictionary(train_voice, sample_rate=args.sample_rate, n_fft=args.n_fft, hop_length=args.hop_length, n_components=args.n_components)

    estimated_bearing, estimated_voice = separate_signals_with_nmf(test_mixture, Basis1, Basis2, sample_rate=args.sample_rate, n_fft=args.n_fft, hop_length=args.hop_length)

    reference = np.stack([test_bearing, test_voice])
    estimated = np.stack([estimated_bearing, estimated_voice])

    plot_time_domain_signals(test_mixture, reference, estimated)
    plot_logmel_spectrogram(test_mixture, title="Mixture Signal")
    plot_logmel_spectrogram(reference[0], title="Reference Bearing Signal")
    plot_logmel_spectrogram(reference[1], title="Reference Voice Signal")
    plot_logmel_spectrogram(estimated[0], title="Estimated Bearing Signal")
    plot_logmel_spectrogram(estimated[1], title="Estimated Voice Signal")

    metrics = calc_bss_metrics(reference, estimated)
    print("BSS Metrics:")
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            formatted = ", ".join([f"{v:.2f}" for v in value])
            print(f"{key}: [{formatted}]")
        else:
            print(f"{key}: {value:.2f}")
    
if __name__ == "__main__":
    main()
