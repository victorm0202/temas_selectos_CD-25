import numpy as np
import pandas as pd
import librosa
from scipy.fft import fft

def generate_sinusoid(dur=1, amp=1, freq=1, phase=0, Fs=100):
    """Generation of sinusoid
    Args:
        dur: Duration (in seconds) of sinusoid (Default value = 1)
        amp: Amplitude of sinusoid (Default value = 1)
        freq: Frequency (in Hertz) of sinusoid (Default value = 1)
        phase: Phase (relative to interval [0,1)) of sinusoid (Default value = 0). 1=2*pi
        Fs: Sampling rate (in samples per second) (Default value = 100)

    Returns:
        x: Signal
        t: Time axis (in seconds)
    """
    num_samples = int(Fs * dur)
    t = np.arange(num_samples) / Fs
    x = amp * np.cos(2 * np.pi * (freq * t + phase))
    return x, t

def generate_example_signal(dur=1, Fs=100):
    """Generate example signal
    Args:
        dur: Duration (in seconds) of signal (Default value = 1)
        Fs: Sampling rate (in samples per second) (Default value = 100)

    Returns:
        x: Signal
        t: Time axis (in seconds)
    """
    N = int(Fs * dur)
    t = np.arange(N) / Fs
    x1 = 1 * np.sin(2 * np.pi * (1.9 * t - 0.3))
    x2 = 0.5 * np.sin(2 * np.pi * (6.1 * t - 0.1))
    x3 = 0.1 * np.sin(2 * np.pi * (20 * t - 0.2))
    y = x1+x2+x3
    return x1, x2, x3, y, t

def sampling_equidistant(x_1, t_1, Fs_2, dur=None):
    """Equidistant sampling of interpolated signal

    Args:
        x_1: Signal to be interpolated and sampled
        t_1: Time axis (in seconds) of x_1
        Fs_2: Sampling rate used for equidistant sampling
        dur: Duration (in seconds) of sampled signal (Default value = None)

    Returns:
        x_2: Sampled signal
        t_2: time axis (in seconds) of sampled signal
    """
    if dur is None:
        dur = len(t_1) * t_1[1]
    N = int(Fs_2 * dur)
    t_2 = np.arange(N) / Fs_2
    x_2 = np.interp(t_2, t_1, x_1)
    return x_2, t_2

def reconstruction_sinc(x, t, t_sinc):
    """Reconstruction from sampled signal using sinc-functions

    Args:
        x: Sampled signal
        t: Equidistant discrete time axis (in seconds) of x
        t_sinc: Equidistant discrete time axis (in seconds) of signal to be reconstructed

    Returns:
        x_sinc: Reconstructed signal having time axis t_sinc
    """
    Fs = 1 / t[1]
    x_sinc = np.zeros(len(t_sinc))
    for n in range(0, len(t)):
        x_sinc += x[n] * np.sinc(Fs * t_sinc - n)
    return x_sinc

def plot_signal_reconstructed(t_1, x_1, t_2, x_2, t_sinc, x_sinc, figsize=(8, 2.2)):
    """Plotting three signals

    Args:
        t_1: Time axis of original signal
        x_1: Original signal
        t_2: Time axis for sampled signal
        x_2: Sampled signal
        t_sinc: Time axis for reconstructed signal
        x_sinc: Reconstructed signal
        figsize: Figure size (Default value = (8, 2.2))
    """
    plt.figure(figsize=figsize)
    plt.plot(t_1, x_1, 'k', linewidth=1, linestyle='dotted', label='Orignal signal')
    plt.stem(t_2, x_2, linefmt='r:', markerfmt='r.', basefmt='None', label='Samples', use_line_collection=True)
    plt.plot(t_sinc, x_sinc, 'b', label='Reconstructed signal')
    plt.title(r'Sampling rate $F_\mathrm{s} = %.0f$' % (1/t_2[1]))
    plt.xlabel('Time (seconds)')
    plt.ylim([-1.8, 1.8])
    plt.xlim([t_1[0], t_1[-1]])
    plt.legend(loc='upper right', framealpha=1)
    plt.tight_layout()
    plt.show()

def f_pitch(p):
    """Compute center frequency for (single or array of) MIDI note numbers
    """
    freq_center = 2 ** ((p - 69) / 12) * 440
    return freq_center

def wave_show(wav_file):
    x, sr = librosa.load(filename)
    plt.figure(figsize=(12, 5))
    plt.ylabel('Amplitude')
    plt.xlabel('Time (seconds)')
    librosa.display.waveshow(x, sr=sr)
    print("Num. muestras:", x.shape, ". Sample rate: ",sr)
    return x, sr

def fft_wav(x, sr):
    X = fft(x)
    X_mag = np.absolute(X)
    f = np.linspace(0, sr, len(X_mag)) # frequency variable
    fft_dataset = pd.DataFrame({'freq': f[:5000], 'mag': X_mag[:5000]})
    fig = px.line(fft_dataset, x='freq', y='mag')
    fig.update_layout(
        autosize=False,
        width=800,
        height=500,
    )
    fig.show()






