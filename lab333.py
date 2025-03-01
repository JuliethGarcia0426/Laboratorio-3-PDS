import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy
from scipy import fftpack
from scipy.signal import butter, filtfilt, wiener
from sklearn.decomposition import FastICA
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play

g1, sr1 = librosa.load('Julieth-García_s-Video-Feb-25_-2025-_2_.wav', sr=None)
g2, sr2 = librosa.load('Julieth-García_s-Video-Feb-25_-2025.wav', sr=None)
g3, sr3 = librosa.load('Julieth-García_s-Video-Feb-25_-2025-_4_.wav', sr=None)
g4, sr4 = librosa.load('Silencio.wav', sr=None)

min_len = min(len(g1), len(g2), len(g3), len(g4))
g1, g2, g3, g4 = g1[:min_len], g2[:min_len], g3[:min_len], g4[:min_len]

for i, (g, sr, title) in enumerate([(g1, sr1, "Micrófono 1"), (g2, sr2, "Micrófono 2"),
                                    (g3, sr3, "Micrófono 3"), (g4, sr4, "Micrófono 4 (Silencio)")]):
    duracion = len(g) / sr  # Tiempo de captura en segundos
    niveles_cuantificacion = 65536  # 16 bits
    print(f"\n🔹 {title}: Frecuencia = {sr} Hz | Duración = {duracion:.2f} s | Cuantización = {niveles_cuantificacion} niveles")
    

def csnr(señal, ruido):
    # Calcular la potencia de la señal y del ruido
    señalp = np.mean(señal**2)
    ruidop = np.mean(ruido**2)
    
    # Calcular el SNR en dB
    snr = 10 * np.log10(señalp / ruidop)
    return snr

for i, g in enumerate([g1, g2, g3], start=1):
    snro = csnr(g, g4)  # Aquí g4 sería tu señal de ruido
    print(f"SNR Original - Micrófono {i}: {snro:.2f} dB")

X = np.c_[g1, g2, g3]
ica = FastICA(n_components=3, random_state=0)
separacion = ica.fit_transform(X)

def pasabanda(señal, sr, lowcut=300, highcut=3400, order=6):
    nyquist = 0.5 * sr  # Frecuencia de Nyquist
    bajo = lowcut / nyquist
    alto = highcut / nyquist
    b, a = butter(order, [bajo, alto], btype='band')
    filtro = filtfilt(b, a, señal)
    return filtro
separacionfiltrada = np.zeros_like(separacion)
for i in range(separacion.shape[1]):
    separacionfiltrada[:, i] = pasabanda(separacion[:, i], sr1) 
    
    



    
    
def normalize_audio(audio, factor=0.9):
    max_val = np.max(np.abs(audio))
    return (audio / max_val) * factor if max_val > 0 else audio  # Escalar entre -1 y 1

for i in range(3):
    audio_final = normalize_audio(separacion[:, i], factor=1.2)
    sf.write(f'fuente_separada_{i+1}_denoised.wav', audio_final, sr1)
    print(f'Archivo fuente_separada_{i+1}_denoised.wav guardado.')


plt.figure(figsize=(10, 7))
for i, (g, sr, title) in enumerate([(g1, sr1, "Micrófono 1"), (g2, sr2, "Micrófono 2"),
                                    (g3, sr3, "Micrófono 3"), (g4, sr4, "Micrófono 4 (Silencio)")]):
    plt.subplot(4, 1, i+1)
    librosa.display.waveshow(g, sr=sr)
    plt.title(f"Señal Original - {title}")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 7))
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(separacion[:, i])
    plt.title(f"Fuente Separada {i+1} (Filtrada y sin ruido)")

plt.tight_layout()
plt.show()


def fft(señal,sr,title):
    N = len(señal)
    T = 1/sr
    y= np.fft.fft(señal)
    x= np.fft.fftfreq(N,T)[:N//2]
    
    plt.plot(x, 2 / N * np.abs(y[:N // 2]))
    plt.grid()
    plt.title(f"Espectro de frecuencias - {title}")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Amplitud")
    plt.show()
    
for i, (g, sr, title) in enumerate([(g1, sr1, "Micrófono 1"), (g2, sr2, "Micrófono 2"),
                                    (g3, sr3, "Micrófono 3"), (g4, sr4, "Micrófono 4 (Silencio)")]):
    print(f"\n🔹 Análisis espectral de la señal - {title}")
    fft(g, sr, title)
    
for i in range(3):
    print(f"\n🔹 Análisis espectral de la fuente separada {i+1}")
    fft(separacionfiltrada[:, i], sr1, f"Fuente Separada {i+1}")    




for i in range(3):
    print(f"Reproduciendo Fuente Separada {i+1} (Mejorada)...")
    audio = AudioSegment.from_wav(f'fuente_separada_{i+1}_denoised.wav')
    play(audio)

print("✅ Proceso finalizado: SNR mejorado, señales filtradas y audios guardados.")