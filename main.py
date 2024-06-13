from comyx.network import (
    UserEquipment,
    BaseStation,
    RIS,
    cascaded_channel_gain,
    effective_channel_gain,
)
from comyx.network import Link
from comyx.propagation import get_noise_power
from comyx.utils import dbm2pow, get_distance

import numpy as np
from numba import jit
from matplotlib import pyplot as plt

plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["figure.figsize"] = (6, 4)

n_antennas = 1  # Number of antennas
n_elements = 70  # Number of elements
realizations = 10000  # Number of Monte Carlo simulations

# shapes
shape_bu = (n_antennas, n_antennas, realizations)
shape_br = (n_elements, n_antennas, realizations)
shape_ru = (n_elements, n_antennas, realizations)
shape_ris = (n_elements, realizations)

# System Parameters
Pt = np.linspace(-45, 20, 80)  # dBm
Pt_lin = dbm2pow(Pt)  # Watt
bandwidth = 1e6  # Bandwidth in Hz
frequency = 2.4e9  # Carrier frequency
temperature = 300  # Kelvin

N0 = get_noise_power(temperature, bandwidth)  # dBm
N0_lin = dbm2pow(N0)  # Watt

nlos_fading_args = {"type": "rayleigh"}  # Rayleigh fading for NLOS
los_fading_args1 = {"type": "rician", "K": 3, "sigma": 1}  # Rician fading for LOS
los_fading_args2 = {"type": "rician", "K": 4, "sigma": 1}  # Rician fading for LOS

pathloss_args1 = {
    "type": "reference",
    "alpha": 3.0,
    "p0": 20,
    "frequency": frequency,
}  # p0 is the reference power in dBm

pathloss_args2 = {
    "type": "reference",
    "alpha": 3.5,
    "p0": 20,
    "frequency": frequency,
}  # p0 is the reference power in dBm

pathloss_args3 = {
    "type": "reference",
    "alpha": 2.7,
    "p0": 20,
    "frequency": frequency,
}  # p0 is the reference power in dBm

# Define Base Stations and Users
BS1 = BaseStation("BS1", position=[0, 0, 10], n_antennas=1, t_power=Pt_lin)
BS2 = BaseStation("BS2", position=[600, 0, 10], n_antennas=1, t_power=Pt_lin)
UEn1 = UserEquipment("UEn1", position=[250, 0, 1], n_antennas=1)
UEn2 = UserEquipment("UEn2", position=[450, 0, 1], n_antennas=1)
UEf = UserEquipment("UEf", position=[300, 0, 1], n_antennas=1)

# Define RIS
R = RIS("RIS", position=[600, 0, 5], n_elements=n_elements)
R.amplitudes = np.ones(shape_ris)  # ideal elements
R.phase_shifts = np.random.uniform(0, 2 * np.pi, shape_ris)

# Display distances
print("Distance between BS1 and UEn1:", get_distance(BS1.position, UEn1.position))
print("Distance between BS2 and UEn2:", get_distance(BS2.position, UEn2.position))
print("Distance between BS1 and UEf:", get_distance(BS1.position, UEf.position))
print("Distance between BS2 and UEf:", get_distance(BS2.position, UEf.position))
print("Distance between RIS and UEn1:", get_distance(R.position, UEn1.position))
print("Distance between RIS and UEn2:", get_distance(R.position, UEn2.position))
print("Distance between RIS and UEf:", get_distance(R.position, UEf.position))

bs1_uen1 = Link(
    BS1,
    UEn1,
    fading_args=los_fading_args1,
    pathloss_args=pathloss_args1,
    shape=shape_bu,
)
bs1_ris = Link(
    BS1, R, fading_args=nlos_fading_args, pathloss_args=pathloss_args1, shape=shape_br
)
ris_uen1 = Link(
    R,
    UEn1,
    fading_args=nlos_fading_args,
    pathloss_args=pathloss_args2,
    shape=shape_ru,
)

bs2_uen2 = Link(
    BS2,
    UEn2,
    fading_args=los_fading_args1,
    pathloss_args=pathloss_args1,
    shape=shape_bu,
)
bs2_ris = Link(
    BS2, R, fading_args=los_fading_args2, pathloss_args=pathloss_args2, shape=shape_br
)
ris_uen2 = Link(
    R,
    UEn2,
    fading_args=nlos_fading_args,
    pathloss_args=pathloss_args2,
    shape=shape_ru,
)

ris_uef = Link(
    R, UEf, fading_args=nlos_fading_args, pathloss_args=pathloss_args2, shape=shape_ris
)


def effective_channel_gain(direct, to_ris, from_ris, ris, n_elements):
    """Calculate the effective channel gain."""
    csc = np.zeros(direct.shape, dtype=np.complex128)

    for i in range(n_elements):
        csc += (
            from_ris.channel_gain[i, :, :]
            * ris.amplitudes[i, :]
            * np.exp(1j * ris.phase_shifts[i, :])
            * to_ris.channel_gain[i, :, :]
        )

    return direct.channel_gain + csc


# Channel Gains
gain_eff1 = effective_channel_gain(bs1_uen1, bs1_ris, ris_uen1, R, n_elements)
gain_eff2 = effective_channel_gain(bs2_uen2, bs2_ris, ris_uen2, R, n_elements)


# Magnitudes
mag_eff1 = np.abs(gain_eff1) ** 2
mag_eff2 = np.abs(gain_eff2) ** 2
mag_d1 = bs1_uen1.magnitude**2
mag_d2 = bs2_uen2.magnitude**2

# SINR and Rate Calculation
UEn1.sinr_wRIS = np.zeros((len(Pt), realizations))
UEn1.sinr_woRIS = np.zeros((len(Pt), realizations))
UEn2.sinr_wRIS = np.zeros((len(Pt), realizations))
UEn2.sinr_woRIS = np.zeros((len(Pt), realizations))
UEf.sinr = np.zeros((len(Pt), realizations))

for i, p in enumerate(Pt_lin):
    p1 = BS1.t_power[i]
    p2 = BS2.t_power[i]

    # SINR with RIS
    UEn1.sinr_wRIS[i, :] = (p1 * mag_eff1) / N0_lin
    UEn2.sinr_wRIS[i, :] = (p2 * mag_eff2) / N0_lin

    # SINR without RIS
    UEn1.sinr_woRIS[i, :] = (p1 * mag_d1) / N0_lin
    UEn2.sinr_woRIS[i, :] = (p2 * mag_d2) / N0_lin

    # Far user receives signals from both base stations
    UEf.sinr[i, :] = (p1 * mag_eff1 + p2 * mag_eff2) / N0_lin

# Rates
rate_wRIS_n1 = np.log2(1 + UEn1.sinr_wRIS)
rate_woRIS_n1 = np.log2(1 + UEn1.sinr_woRIS)
rate_wRIS_n2 = np.log2(1 + UEn2.sinr_wRIS)
rate_woRIS_n2 = np.log2(1 + UEn2.sinr_woRIS)
rate_f = np.log2(1 + UEf.sinr)

# Thresholds
thresh_n1 = 1
thresh_n2 = 1
thresh_f = 1


# Outage Calculation
@jit(nopython=True)
def get_outage(rate, thresh):
    outage = np.zeros((len(Pt), 1))

    for i in range(len(Pt)):
        for k in range(realizations):
            if rate[i, k] < thresh:
                outage[i] += 1

    return outage


UEn1.outage_wRIS = get_outage(rate_wRIS_n1, thresh_n1) / realizations
UEn1.outage_woRIS = get_outage(rate_woRIS_n1, thresh_n1) / realizations
UEn2.outage_wRIS = get_outage(rate_wRIS_n2, thresh_n2) / realizations
UEn2.outage_woRIS = get_outage(rate_woRIS_n2, thresh_n2) / realizations
UEf.outage_ = get_outage(rate_f, thresh_f) / realizations

# Average Rates
UEn1.rate_wRIS = np.mean(rate_wRIS_n1, axis=-1)
UEn1.rate_woRIS = np.mean(rate_woRIS_n1, axis=-1)
UEn2.rate_wRIS = np.mean(rate_wRIS_n2, axis=-1)
UEn2.rate_woRIS = np.mean(rate_woRIS_n2, axis=-1)
UEf.rate_ = np.mean(rate_f, axis=-1)

# Plotting
plot_args = {
    "markevery": 10,
    "color": "k",
    "markerfacecolor": "r",
}

plt.figure()
plt.plot(
    Pt, UEn1.rate_wRIS, label="Rate Near User 1 (with RIS)", marker="s", **plot_args
)
plt.plot(
    Pt, UEn1.rate_woRIS, label="Rate Near User 1 (without RIS)", marker="d", **plot_args
)
plt.plot(
    Pt, UEn2.rate_wRIS, label="Rate Near User 2 (with RIS)", marker="o", **plot_args
)
plt.plot(
    Pt, UEn2.rate_woRIS, label="Rate Near User 2 (without RIS)", marker="v", **plot_args
)

plt.plot(Pt, UEf.rate, label="Rate Far User", marker="x", **plot_args)
plt.xlabel("Transmit power (dBm)")
plt.ylabel("Rate (bps/Hz)")
plt.grid(alpha=0.25)
plt.legend()
plt.savefig("noma_ris_rate.png", dpi=300, bbox_inches="tight")
plt.close()