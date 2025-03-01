# LABORATORIO 3 ~ EL PROBLEMA DEL COCTEL

## ***Descripción del caso***

En un evento tipo coctel, se instalaron varios micrófonos para escuchar lo que las personas estaban hablando; una vez terminó la fiesta, se solicitó a los ingenieros que entregaran el audio de la voz de uno de los participantes.

## ***Configuración del sistema***

En un área cuadrada de 4 metros por 4 metros, se han ubicado tres micrófonos y tres fuentes sonoras, cada una con una posición específica y una dirección determinada. Los micrófonos se identifican como Micrófono 1, Micrófono 2 y Micrófono 3, mientras que las fuentes sonoras se denominan Fuente 1, Fuente 2 y Fuente 3, Esta descripción esta representada en un esquema:

![image](https://github.com/user-attachments/assets/4fc4bcd5-e2a5-43c5-a267-802ffa196e23)


Cada fuente emite sonido en una dirección particular: Fuente 1 está orientada hacia el sur, Fuente 2 hacia el norte y Fuente 3 hacia el este. Las distancias entre las fuentes y los micrófonos son las siguientes:

Fuente 1 se encuentra a 1.15 metros de Micrófono 3, 2.53 metros de Micrófono 2 y 2.16 metros de Micrófono 1.

Fuente 2 está ubicada a 1.14 metros de Micrófono 1, 2.88 metros de Micrófono 2 y 4.39 metros de Micrófono 3.

Fuente 3 tiene una distancia de 1.35 metros respecto a Micrófono 2, 1.12 metros respecto a Micrófono 3 y 3.04 metros respecto a Micrófono 1.

## ***Digitalización del sistema***

![image](https://github.com/user-attachments/assets/7cf6354a-b888-4f66-907f-5678d8c0706b)



- *Frecuencia de muestreo* 

```python

g1, sr1 = librosa.load('Julieth-García\_s-Video-Feb-25\_-2025-\_2\_.wav', sr=None)

```

La frecuencia de muestreo **no está fijada manualmente** (sr=None mantiene la original del archivo de audio).  

- *Niveles de cuantificación*

```python

niveles\_cuantificacion = 65536 # 16 bits

```

Esto indica que las señales están cuantizadas con 16 bits por muestra, 

- *Tiempo de captura*

En cada audio se utilizaron 15 segundos de grabación, sin embargo en el código se uso la instrucción 

```python

duracion = len(g) / sr

```

*Y la instrucción* 

```python

min\_len = min(len(g1), len(g2), len(g3), len(g4))

` `g1, g2, g3, g4 = g1[:min\_len], g2[:min\_len], g3[:min\_len], g4[:min\_len]

```

Para recortar las señales a la misma longitud y evitar desajustes 

- *Relación Señal-Ruido*

```python

def csnr(señal, ruido): 

señalp = np.mean(señal\*\*2) 

ruidop = np.mean(ruido\*\*2)

` `snr = 10 \* np.log10(señalp / ruidop) 

return snr

```

El **SNR original** de cada micrófono se calcula comparando la señal con g4 (el ruido de referencia).

## Descripción del código

 ### **Importación de Librerías**

```python

import numpy as np

import matplotlib.pyplot as plt

import librosa 

import librosa.display 

import scipy from scipy 

import fftpack from scipy.signal 

import butter, filtfilt, wiener from sklearn.decomposition 

import FastICA 

import soundfile as sf from pydub 

import AudioSegment from pydub.playback 

import play

```

*¿Qué hace cada librería?*

- numpy: Manejo de arreglos numéricos.
- matplotlib.pyplot: Graficación de señales y espectros.
- librosa: Procesamiento de audio.
- scipy.signal: Filtrado de señales.
- sklearn.decomposition.FastICA: Algoritmo de separación de fuentes.
- soundfile: Lectura y escritura de archivos de audio.
- pydub: Reproducción de audio.
- 
###  **Carga de Señales de Audio**

```python

g1, sr1 = librosa.load('Julieth-García\_s-Video-Feb-25\_-2025-\_2\_.wav', sr=None)

g2, sr2 = librosa.load('Julieth-García\_s-Video-Feb-25\_-2025.wav', sr=None)

g3, sr3 = librosa.load('Julieth-García\_s-Video-Feb-25\_-2025-\_4\_.wav', sr=None)

g4, sr4 = librosa.load('Silencio.wav', sr=None)

```

*Se cargan 4 archivos de audio:*

- g1, g2, g3: Señales de 3 micrófonos.
- g4: Archivo de **ruido de referencia**.

sr=None mantiene la tasa de muestreo original de cada archivo.

### **Ajuste de Longitud**

```python

min\_len = min(len(g1), len(g2), len(g3), len(g4))

g1, g2, g3, g4 = g1[:min\_len], g2[:min\_len], g3[:min\_len], g4[:min\_len]

```

Se recortan todas las señales a la misma longitud mínima para evitar desajustes.

### **Información de las Señales**

```python

for i, (g, sr, title) in enumerate([(g1, sr1, "Micrófono 1"), (g2, sr2, "Micrófono 2"),

`                                    `(g3, sr3, "Micrófono 3"), (g4, sr4, "Micrófono 4 (Silencio)")]):

`    `duracion = len(g) / sr  

`    `niveles\_cuantificacion = 65536  # 16 bits

`    `print(f"\n🔹 {title}: Frecuencia = {sr} Hz | Duración = {duracion:.2f} s | Cuantización = {niveles\_cuantificacion} niveles")

```

*Se calcula:*

- Frecuencia de muestreo (sr).
- Duración de la señal en segundos.
- Niveles de cuantificación (16 bits = 65536 niveles).

  ### **Cálculo del SNR (Relación Señal/Ruido)**

```python

def csnr(señal, ruido):

`    `señalp = np.mean(señal\*\*2)  

`    `ruidop = np.mean(ruido\*\*2)  

`    `snr = 10 \* np.log10(señalp / ruidop)  

`    `return snr

for i, g in enumerate([g1, g2, g3], start=1):

`    `snro = csnr(g, g4)  # Se usa g4 (ruido de referencia)

`    `print(f"SNR Original - Micrófono {i}: {snro:.2f} dB")

```

Se imprime el SNR original de cada micrófono**.**

### **Separación de Fuentes con FastICA**

```python

X = np.c\_[g1, g2, g3]

ica = FastICA(n\_components=3, random\_state=0)

separacion = ica.fit\_transform(X)

```

Se combinan las señales (X) y se aplica FastICA para separar las fuentes.

n\_components=3 indica que queremos 3 fuentes independientes.

### **Filtrado Pasabanda (300 Hz - 3400 Hz)**

```python

def pasabanda(señal, sr, lowcut=300, highcut=3400, order=6):

`    `nyquist = 0.5 \* sr  

`    `bajo = lowcut / nyquist

`    `alto = highcut / nyquist

`    `b, a = butter(order, [bajo, alto], btype='band')

`    `filtro = filtfilt(b, a, señal)

`    `return filtro

separacionfiltrada = np.zeros\_like(separacion)

for i in range(separacion.shape[1]):

`    `separacionfiltrada[:, i] = pasabanda(separacion[:, i], sr1)  

```

Se usa un filtro pasabanda Butterworth para eliminar frecuencias fuera del rango de voz.

### **Normalización y Guardado de Audio**

```python

def normalize\_audio(audio, factor=0.9):

`    `max\_val = np.max(np.abs(audio))

`    `return (audio / max\_val) \* factor if max\_val > 0 else audio  

for i in range(3):

`    `audio\_final = normalize\_audio(separacion[:, i], factor=1.2)

`    `sf.write(f'fuente\_separada\_{i+1}\_denoised.wav', audio\_final, sr1)

`    `print(f'Archivo fuente\_separada\_{i+1}\_denoised.wav guardado.')

```

Normaliza las señales separadas y las guarda en archivos .wav.

### **Visualización de Señales Originales**

```python

plt.figure(figsize=(10, 7))

for i, (g, sr, title) in enumerate([(g1, sr1, "Micrófono 1"), (g2, sr2, "Micrófono 2"),

`                                    `(g3, sr3, "Micrófono 3"), (g4, sr4, "Micrófono 4 (Silencio)")]):

`    `plt.subplot(4, 1, i+1)

`    `librosa.display.waveshow(g, sr=sr)

`    `plt.title(f"Señal Original - {title}")

plt.tight\_layout()

plt.show()

```

![image](https://github.com/user-attachments/assets/0d4f6205-569d-4263-b55a-94567c540096)


Se grafican las** ondas de las señales originales**.**

### **Visualización de señales separadas**

```python

plt.figure(figsize=(10, 7))

for i in range(3):

`    `plt.subplot(3, 1, i+1)

`    `plt.plot(separacion[:, i])

`    `plt.title(f"Fuente Separada {i+1} (Filtrada y sin ruido)")

plt.tight\_layout()

plt.show()

```

![image](https://github.com/user-attachments/assets/3f59aad3-abe9-46e2-a531-28c75aa4bca8)


Muestra las fuentes separadas con ICA, después del filtrado.

### **Análisis en Frecuencia (FFT)**

```python

def fft(señal, sr, title):

`    `N = len(señal)

`    `T = 1 / sr

`    `y = np.fft.fft(señal)

`    `x = np.fft.fftfreq(N, T)[:N // 2]



`    `plt.plot(x, 2 / N \* np.abs(y[:N // 2]))

`    `plt.grid()

`    `plt.title(f"Espectro de frecuencias - {title}")

`    `plt.xlabel("Frecuencia (Hz)")

`    `plt.ylabel("Amplitud")

`    `plt.show()

```

![image](https://github.com/user-attachments/assets/ce3427a6-c812-483c-9ed8-80cc14cc49a9)
![image](https://github.com/user-attachments/assets/de9ac850-c308-4810-8a17-f62d5142f06e)
![image](https://github.com/user-attachments/assets/719981d4-5c2b-4f8e-ab44-498182c838af)
![image](https://github.com/user-attachments/assets/76139c1f-5ca2-465e-af7f-205679dc4384)


```python

for i in range(3):

`    `print(f"\n Análisis espectral de la fuente separada {i+1}")

`    `fft(separacionfiltrada[:, i], sr1, f"Fuente Separada {i+1}")

```

![image](https://github.com/user-attachments/assets/e6877fb5-918b-40e1-8845-def7c898e742)
![image](https://github.com/user-attachments/assets/5c252a65-9176-4458-82f6-ddbb8abe6017)
![image](https://github.com/user-attachments/assets/3098ac43-956b-4933-80ea-0ca4602a0471)



### **Reproducción de las Fuentes Separadas**

```python

for i in range(3):

`    `print(f"Reproduciendo Fuente Separada {i+1} (Mejorada)...")

`    `audio = AudioSegment.from\_wav(f'fuente\_separada\_{i+1}\_denoised.wav')

`    `play(audio)

```

- Carga los archivos .wav generados con pydub.
- Reproduce cada señal separada.

## Análisis del SNR

#### Los valores de SNR original obtenidos para cada micrófono son:

- Micrófono 1: 20.74 dB
- Micrófono 2: 19.14 dB
- Micrófono 3: 23.29 dB

Recordemos que la relación señal/ruido (SNR) mide la proporción entre la potencia de la señal y la potencia del ruido en decibeles (dB).

#### Interpretación de los valores de SNR:
- SNR alto (mayor a 20 dB) → Buena calidad, el ruido es bajo en comparación con la señal.
- SNR medio (15-20 dB) → La señal es clara, pero el ruido aún es perceptible.
- SNR bajo (menor a 15 dB) → La señal está muy contaminada por ruido.

  #### Análisis por micrófono
- Micrófono 1 - SNR = 20.74 dB

Calidad: Moderadamente buena.

Ruido presente, pero no dominante.

Posible causa de ruido: Puede haber interferencias ambientales o ruido de fondo.

- Micrófono 2 - SNR = 19.14 dB

Calidad: Aceptable, pero con más ruido que el Micrófono 1.

Ruido más perceptible, podría afectar la inteligibilidad del audio.

Posible causa: Mayor sensibilidad al ruido o posición diferente del micrófono.

- Micrófono 3 - SNR = 23.29 dB

Calidad: Mejor que los otros dos.

Señal más clara y menos ruido presente.

Posible causa: Mejor posicionamiento del micrófono, mejor captación o menor interferencia.

## Audios

**[🔊 Descargar audio fuente 1](https://github.com/JuliethGarcia0426/Laboratorio-3-PDS/raw/refs/heads/main/Julieth-Garc%C3%ADa_s-Video-Feb-25_-2025-_2_.wav)**

**[🔊 Descargar audio fuente 2](https://github.com/JuliethGarcia0426/Laboratorio-3-PDS/raw/refs/heads/main/Julieth-Garc%C3%ADa_s-Video-Feb-25_-2025.wav)**

**[🔊 Descargar audio fuente 3](https://github.com/JuliethGarcia0426/Laboratorio-3-PDS/raw/refs/heads/main/Julieth-Garc%C3%ADa_s-Video-Feb-25_-2025-_4_.wav)**

**[🔊 Descargar audio silencio](https://github.com/JuliethGarcia0426/Laboratorio-3-PDS/raw/refs/heads/main/Silencio.wav)**

**[🔊 Descargar audio fuente separada 1](https://github.com/JuliethGarcia0426/Laboratorio-3-PDS/raw/refs/heads/main/fuente_separada_1_denoised.wav)**

**[🔊 Descargar audio fuente separada 2](https://github.com/JuliethGarcia0426/Laboratorio-3-PDS/raw/refs/heads/main/fuente_separada_2_denoised.wav)**

**[🔊 Descargar audio fuente separada 3](https://github.com/JuliethGarcia0426/Laboratorio-3-PDS/raw/refs/heads/main/fuente_separada_3_denoised.wav)**




