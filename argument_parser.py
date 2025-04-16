import argparse

def get_args():
    parser = argparse.ArgumentParser(description="NMF Source Separation Configuration")

    # 파일 경로
    parser.add_argument('--voice_path', type=str, default="Data/GD.mp3", help='Voice signal file path')
    parser.add_argument('--bearing_normal_path', type=str, default="Data/PUMP-N.mp3", help='Bearing normal signal file path')
    parser.add_argument('--bearing_fault_path', type=str, default="Data/PUMP-IR.mp3", help='Bearing fault signal file path')

    parser.add_argument('--sample_rate', type=int, default=44100, help='Sampling rate')
    parser.add_argument('--n_fft', type=int, default=2048, help='FFT window size')
    parser.add_argument('--hop_length', type=int, default=512, help='STFT hop length')
    parser.add_argument('--n_components', type=int, default=128, help='Number of NMF bases')


    return parser.parse_args()
