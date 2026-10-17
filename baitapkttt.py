import numpy as np
import matplotlib.pyplot as plt

mu = 0
sigma = 1
N = 100000

X_gaussian = sigma * np.random.randn(N) + mu
X_exponential = np.random.exponential(scale=1, size=N) - 1
X_laplace = np.random.laplace(loc=0, scale=1/np.sqrt(2), size=N) # Căn chỉnh scale để Var = 1

# Thiết lập đồ thị
fig, axs = plt.subplots(3, 1, figsize=(10, 12))
fig.tight_layout(pad=5.0)

# 1. Vẽ PDF thực nghiệm
bins = 100
axs[0].hist(X_gaussian, bins=bins, density=True, alpha=0.5, label='Gaussian')
axs[0].hist(X_exponential, bins=bins, density=True, alpha=0.5, label='Exponential (shifted)')
axs[0].hist(X_laplace, bins=bins, density=True, alpha=0.5, label='Laplace')
axs[0].set_title('1. Hàm mật độ xác suất thực nghiệm (PDF)')
axs[0].set_xlim([-5, 5])
axs[0].legend()
axs[0].grid(True)

# 3. Tính và vẽ Hàm tự tương quan
# Cắt lấy 100 lag đầu tiên để dễ quan sát
lags = np.arange(-50, 51)
def calc_autocorr(x, lags):
    res = [np.correlate(x, np.roll(x, lag))[0]/N for lag in lags]
    return res

axs[1].stem(lags, calc_autocorr(X_gaussian, lags), linefmt='b-', markerfmt='bo', basefmt='k-', label='Gaussian')
axs[1].stem(lags, calc_autocorr(X_exponential, lags), linefmt='g-', markerfmt='go', basefmt='k-', label='Exponential')
axs[1].stem(lags, calc_autocorr(X_laplace, lags), linefmt='r-', markerfmt='ro', basefmt='k-', label='Laplace')
axs[1].set_title('3. Hàm tự tương quan (Autocorrelation) tại các lag nhỏ')
axs[1].set_xlabel('Lag')
axs[1].legend()
axs[1].grid(True)


# 4. Tính và vẽ PSD sử dụng fft và fftshift
def calc_psd(x):
    X_fft = np.fft.fftshift(np.fft.fft(x))
    psd = (np.abs(X_fft) ** 2) / N
    return psd

freqs = np.fft.fftshift(np.fft.fftfreq(N))

axs[2].plot(freqs, calc_psd(X_gaussian), alpha=0.7, label='Gaussian')
axs[2].plot(freqs, calc_psd(X_exponential), alpha=0.7, label='Exponential')
axs[2].plot(freqs, calc_psd(X_laplace), alpha=0.7, label='Laplace')
axs[2].set_title('4. Mật độ phổ công suất (PSD)')
axs[2].set_xlabel('Normalized Frequency')
axs[2].set_ylim([0, 5])
axs[2].legend()
axs[2].grid(True)

plt.show()