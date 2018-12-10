import soundfile as sf
import numpy as np
from scipy import fftpack
import cv2 as cv
import matplotlib.animation as anim
import matplotlib.pyplot as plt

plt.interactive(False)


# from scipy import signal
# from IPython.display import Image, display


def plot_wav(filename, start_time, end_time):
    samples, samplerate = sf.read(filename)
    samples_left = samples[int(start_time * samplerate):int(end_time * samplerate), 0] + 0.5
    samples_right = samples[int(start_time * samplerate):int(end_time * samplerate), 1] + 0.5
    samples_2d = samples_left[np.newaxis, :] * samples_right[:, np.newaxis]
    samples_2d_fft = fftpack.fftshift(fftpack.fftn(samples_2d))
    # print(np.shape(samples_2d))
    plt.interactive(False)
    # print(samples_2d)
    image = np.log(np.abs(samples_2d_fft)) + np.angle(samples_2d_fft)
    kernel = np.ones((10, 10), np.float32) / 100
    image = cv.filter2D(image, -1, kernel)
    plt.matshow(image, cmap=plt.cm.hot, animated=True)
    # plt.colorbar()
    plt.show()


plot_wav('tatran.wav', 14, 14.01)

# Writer = anim.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# fig = plt.figure()
#
# frames = []
# for i in range(10):
#     frame = plot_wav('tatran.wav', 20 + (i / 100.0), (20 + 0.005) + (i / 100.0))
#     frames.append([frame])
# my_animation = anim.ArtistAnimation(fig, frames, interval=50, blit=True,
#                                     repeat_delay=1000)
# my_animation.save('tatran_animation.mp4', writer=writer)
