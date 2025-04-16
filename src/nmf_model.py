import numpy as np
import librosa
from sklearn.decomposition import NMF


def train_nmf_dictionary(
    signal: np.ndarray,
    sample_rate: int = 44100,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_components: int = 64,
    max_iter: int = 2000,
    tol: float = 1e-4,
    random_state: int = 0
) -> np.ndarray:

    # Magnitude spectrogram
    stft_matrix = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    magnitude_spectrogram = np.abs(stft_matrix)

    # NMF 모델 학습
    nmf_model = NMF(
        n_components=n_components,
        init='random',
        random_state=random_state,
        max_iter=max_iter,
        solver='mu',
        beta_loss='frobenius',
        tol=tol
    )
    nmf_model.fit(magnitude_spectrogram.T)  # shape = (time, freq)
    
    # 학습된 사전(Components) 반환
    return nmf_model.components_


def separate_signals_with_nmf(
    mixed_signal: np.ndarray,
    dict_source1: np.ndarray,
    dict_source2: np.ndarray,
    sample_rate: int = 44100,
    n_fft: int = 1024,
    hop_length: int = 512,
    max_iter: int = 2000,
    tol: float = 1e-4,
    random_state: int = 0
) -> tuple[np.ndarray, np.ndarray]:

    # STFT (복소수 형태), magnitude 추출
    stft_mixed = librosa.stft(mixed_signal, n_fft=n_fft, hop_length=hop_length)
    mag_mixed = np.abs(stft_mixed).T.astype(np.float32)  # shape=(time, freq)

    # 두 사전 합치기
    dictionary_total = np.vstack([dict_source1, dict_source2]).astype(np.float32)
    n_total_components = dictionary_total.shape[0]  # 두 사전의 총 컴포넌트 수

    # NMF 모델 생성 (Custom init)
    nmf_model = NMF(
        n_components=n_total_components,
        init='custom',
        solver='mu',
        beta_loss='frobenius',
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    )

    # W_init: (time, n_total_components)
    # H_init: (n_total_components, freq)
    W_init = np.abs(np.random.randn(mag_mixed.shape[0], n_total_components)).astype(np.float32)
    H_init = dictionary_total.copy()

    # NMF 분해 (W: (time, n_total_components), H: (n_total_components, freq))
    activation_matrix = nmf_model.fit_transform(mag_mixed, W=W_init, H=H_init)

    # 추정된 W를 source1, source2에 해당하는 부분으로 분리
    n_comp_s1 = dict_source1.shape[0]
    activation_s1 = activation_matrix[:, :n_comp_s1]
    activation_s2 = activation_matrix[:, n_comp_s1:]

    # 각 소스의 STFT를 magnitude 스펙트럼으로 추정
    s1_hat = activation_s1 @ dict_source1  # shape = (time, freq)
    s2_hat = activation_s2 @ dict_source2  # shape = (time, freq)

    # 두 소스 추정량의 합이 0이 되지 않도록 epsilon 추가
    total_hat = s1_hat + s2_hat + 1e-8

    # 마스크 계산 (shape를 복원 위해 Transpose)
    mask_s1 = (s1_hat / total_hat).T
    mask_s2 = (s2_hat / total_hat).T

    # 복원된 STFT (위상은 원본 mixed 신호의 것 사용)
    stft_source1 = mask_s1 * stft_mixed
    stft_source2 = mask_s2 * stft_mixed

    # ISTFT로 시간 도메인 신호 복원
    estimated_source1 = librosa.istft(stft_source1, hop_length=hop_length, length=len(mixed_signal))
    estimated_source2 = librosa.istft(stft_source2, hop_length=hop_length, length=len(mixed_signal))

    return estimated_source1, estimated_source2
