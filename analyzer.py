import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.signal
import soundfile
from scipy.fft import fft, fftfreq
from typing import Union

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_DISPLAY_CEILING_DB: float = -10.0

# Threshold table used by both plot_spectrogram and plot_spectrum to choose
# sensible major/minor x-axis tick spacing based on the visible frequency range.
# Each row is (max_range_hz, major_tick_hz, minor_tick_hz).
TICK_THRESHOLDS: list[tuple[float, float, float]] = [
    (8,      0.1,   0.01),
    (80,     1,     0.1),
    (800,    10,    1),
    (8000,   100,   10),
    (np.inf, 1000,  100),  # fallback for > 8000 Hz
]

FIG_SIZE = (16, 10)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse_time(t: Union[float, int, str]) -> float:
    """Convert a time value to fractional minutes.

    Accepted formats
    ----------------
    - ``float`` / ``int``  – interpreted as **seconds** (e.g. ``90`` → 1.5 min)
    - ``str`` ``"M:SS"`` or ``"MM:SS"``  – e.g. ``"1:30"``  → 1.5 min
    - ``str`` of a bare number – interpreted as seconds  (e.g. ``"90"``)

    Returns
    -------
    float
        Time expressed as fractional minutes.
    """
    if isinstance(t, str):
        if ":" in t:
            parts = t.split(":", 1)
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes + seconds / 60.0
        else:
            return float(t) / 60.0
    # numeric → seconds
    return float(t) / 60.0


def _fmt_minutes(minutes: float) -> str:
    """Format fractional minutes as ``M:SS`` for axis tick labels."""
    total_seconds = round(minutes * 60)
    m, s = divmod(total_seconds, 60)
    return f"{m}:{s:02d}"


class SpectrumAnalyzer:
    """Load a stereo audio file and expose spectral analysis / visualisation.

    Parameters
    ----------
    path:
        Path to any audio file supported by *libsndfile* (WAV, FLAC, AIFF, …).

    Attributes
    ----------
    samples : numpy.ndarray, shape (N, 2)
        Raw PCM samples for both channels.
    fs : int
        Sample rate in Hz.
    left_channel : numpy.ndarray
        Convenience view of the left (index-0) channel.
    right_channel : numpy.ndarray
        Convenience view of the right (index-1) channel.
    """

    def __init__(self, path: str) -> None:
        self.samples, self.fs = soundfile.read(path)
        if self.samples.ndim == 1:
            # Promote mono to two identical channels so the rest of the code
            # works without special-casing.
            self.samples = np.stack([self.samples, self.samples], axis=1)
        self.left_channel: np.ndarray = self.samples[:, 0]
        self.right_channel: np.ndarray = self.samples[:, 1]

        # Populated by fft(); consumed by plot_spectrum().
        self._Z: list[np.ndarray] = []
        self._f: list[np.ndarray] = []
        self._steps_per_hz: list[float] = []

    # ------------------------------------------------------------------
    # Public analysis methods
    # ------------------------------------------------------------------

    def plot_spectrogram(
        self,
        start: Union[float, int, str],
        end: Union[float, int, str],
        window_size: int | None = None,
        display_ceiling_db: float = DEFAULT_DISPLAY_CEILING_DB,
    ) -> None:
        """Plot a short-time Fourier-transform spectrogram for both channels.

        Parameters
        ----------
        start:
            Start of the analysis window.  Accepts seconds (``int``/``float``),
            a ``"M:SS"`` string, or a bare numeric string (seconds).
        end:
            End of the analysis window.  Same format as *start*.
        window_size:
            STFT window length in samples.  Defaults to ``fs // 4``.
        display_ceiling_db:
            Upper bound of the colour scale in dBFS.  Defaults to
            ``DEFAULT_DISPLAY_CEILING_DB`` (``-10``).
        """
        start_min = _parse_time(start)
        end_min   = _parse_time(end)

        if window_size is None:
            window_size = self.fs // 4

        start_sample = int(self.fs * 60 * start_min)
        end_sample   = int(self.fs * 60 * end_min)

        fig, ax = plt.subplots(2, figsize=FIG_SIZE, sharey=True, sharex=True)
        fig.supylabel("Frequency [Hz]", x=0)
        fig.supxlabel("Time [M:SS]", y=0)
        plt.tight_layout()

        for i, channel_name, channel_samples in zip(
            (0, 1),
            ("Left channel", "Right channel"),
            (self.left_channel, self.right_channel),
        ):
            chunk = channel_samples[start_sample:end_sample]
            f, t, Zxx = scipy.signal.spectrogram(
                chunk,
                self.fs,
                scaling="spectrum",
                mode="magnitude",
                nperseg=window_size,
                noverlap=window_size // 2,
            )
            magnitude = 10 * np.log10(Zxx)

            ax[i].pcolormesh(
                t / 60 + start_min,
                f,
                magnitude,
                cmap="inferno",
                vmin=-60,
                vmax=display_ceiling_db,
            )
            ax[i].set_title(channel_name)
            ax[i].set_yscale("log")
            ax[i].set_ylim(20, 20000)
            ax[i].set_xlim(start_min, end_min)

            freq_ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
            ax[i].set_yticks(freq_ticks)
            ax[i].yaxis.set_major_formatter(ticker.ScalarFormatter())
            ax[i].yaxis.get_major_formatter().set_scientific(False)
            ax[i].yaxis.get_major_formatter().set_useOffset(False)

            ax[i].xaxis.set_major_locator(plt.MultipleLocator(1))
            ax[i].xaxis.set_minor_locator(plt.MultipleLocator(1 / 6))

            # Replace numeric minute labels with M:SS strings.
            ax[i].xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda x, _: _fmt_minutes(x))
            )
            ax[i].tick_params(axis="x", rotation=90)

        plt.show()

    def fft(
        self,
        start: Union[float, int, str],
        end: Union[float, int, str],
    ) -> None:
        """Compute the one-sided magnitude spectrum for both channels.

        Results are stored in private attributes and consumed by
        :meth:`plot_spectrum`.  Call this method before calling
        :meth:`plot_spectrum`.

        The spectrum is scaled so that a full-scale sine wave reads 0 dBFS
        (``norm="forward"`` on the FFT, then ×2 for the one-sided fold,
        with the DC bin and — for even-length signals — the Nyquist bin
        corrected back to ×1 because they have no negative-frequency mirror).

        Parameters
        ----------
        start:
            Start of the analysis window.  Accepts seconds (``int``/``float``),
            a ``"M:SS"`` string, or a bare numeric string (seconds).
        end:
            End of the analysis window.  Same format as *start*.
        """
        start_min = _parse_time(start)
        end_min   = _parse_time(end)

        start_sample = int(self.fs * 60 * start_min)
        end_sample   = int(self.fs * 60 * end_min)

        self._Z = []
        self._f = []
        self._steps_per_hz = []

        for channel_samples in (self.left_channel, self.right_channel):
            chunk = channel_samples[start_sample:end_sample]
            N = len(chunk)

            spectrum = fft(chunk, norm="forward")[: N // 2] * 2
            # DC bin (0 Hz) has no negative-frequency mirror → undo the ×2.
            spectrum[0] /= 2
            # Nyquist bin only exists for even N and also has no mirror.
            if N % 2 == 0:
                spectrum[-1] /= 2

            self._Z.append(spectrum)
            self._f.append(fftfreq(N, 1.0 / self.fs)[: N // 2])
            self._steps_per_hz.append(N / self.fs)

    def plot_spectrum(
        self,
        min_f: float = 20,
        max_f: float = 1000,
        log_y: bool = True,
        display_ceiling_db: float = DEFAULT_DISPLAY_CEILING_DB,
        min_peak_dist: float | None = None,
        min_peak_level: float | None = None,
    ) -> plt.Figure:
        """Plot the FFT spectrum computed by :meth:`fft`.

        Must be preceded by a call to :meth:`fft`.

        Parameters
        ----------
        min_f:
            Lower bound of the displayed frequency range in Hz.
        max_f:
            Upper bound of the displayed frequency range in Hz.
        log_y:
            If ``True``, the y-axis is in dBFS; otherwise linear magnitude.
        display_ceiling_db:
            Upper y-axis limit in dBFS (only used when *log_y* is ``True``).
            Defaults to ``DEFAULT_DISPLAY_CEILING_DB`` (``-10``).
        min_peak_dist:
            Minimum separation between detected peaks in Hz.  ``None`` (default)
            uses an automatic value of ``(max_f - min_f) / 200``.
        min_peak_level:
            Peaks below this dBFS threshold are suppressed.  ``None`` disables
            threshold filtering.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object for further customisation or saving.
        """
        fig, ax = plt.subplots(2, figsize=FIG_SIZE, sharey=True, sharex=True)
        fig.supylabel("Magnitude [dBFS]" if log_y else "Magnitude [linear]", x=0)
        fig.supxlabel("Frequency [Hz]", y=0)
        fig.tight_layout()

        freq_range = max_f - min_f
        if min_peak_dist is None:
            min_peak_dist = freq_range / 200

        for i, (channel_name, Z, f, steps_per_hz) in enumerate(
            zip(
                ("Left channel", "Right channel"),
                self._Z,
                self._f,
                self._steps_per_hz,
            )
        ):
            f_mask = (f >= min_f) & (f <= max_f)
            f_vis = f[f_mask]
            Z_vis = Z[f_mask]

            magnitude = np.abs(Z_vis).copy()  # copy avoids mutating self._Z
            phase = np.angle(Z_vis)

            if log_y:
                magnitude = 20 * np.log10(magnitude)
                ax[i].set_ylim(-120, display_ceiling_db)

            ax[i].set_title(channel_name)
            ax[i].plot(f_vis, magnitude, linewidth=0.25)
            ax[i].set_xlim(min_f, max_f)
            ax[i].grid(True, which="major", linestyle="-", linewidth=0.5)
            ax[i].grid(True, which="minor", linestyle="--", linewidth=0.25)
            ax[i].tick_params(axis="x", rotation=90)
            ax[i].minorticks_on()

            # --- x-axis tick spacing ----------------------------------------
            _, major_base, minor_base = next(
                row for row in TICK_THRESHOLDS if freq_range <= row[0]
            )
            ax[i].xaxis.set_major_locator(plt.MultipleLocator(base=major_base))
            ax[i].xaxis.set_minor_locator(plt.MultipleLocator(base=minor_base))

            # --- peak detection ---------------------------------------------
            peaks, _ = scipy.signal.find_peaks(
                magnitude, distance=min_peak_dist * steps_per_hz
            )
            if min_peak_level is not None:
                peaks = peaks[magnitude[peaks] > min_peak_level]

            ax[i].plot(f_vis[peaks], magnitude[peaks], ".")
            for peak in peaks:
                ax[i].annotate(
                    f"{f_vis[peak]:.2f}@{magnitude[peak]:.1f}@{phase[peak] / np.pi:.2f}",
                    xy=(f_vis[peak], magnitude[peak]),
                    xytext=(-0.5, 0.75),
                    textcoords="offset fontsize",
                    rotation=90,
                    fontsize=6,
                )

        plt.show()
