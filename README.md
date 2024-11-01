## **Performance Analysis of Aerial RIS-Assisted CoMP-NOMA Networks**

This repository contains the code and data for the paper, *Machine Learning-Driven Performance Analysis of Compressed Communication in Aerial-RIS Networks for Future 6G Networks*, which explores an integrated approach for enhancing network capacity, coverage, and spectral efficiency by combining Reconfigurable Intelligent Surfaces (RIS), Coordinated Multipoint Transmission (CoMP), and Non-Orthogonal Multiple Access (NOMA). The study focuses on a network setup with Unmanned Aerial Vehicles (UAV)-mounted RIS to provide dynamic coverage for dense urban and 6G applications.

### Repository Structure

The structure of the repository is as follows:

- **`core/`**: Core functionalities including signal processing and RIS configurations.
- **`fading/`**: Contains scripts to simulate Rayleigh and Rician fading for different channel environments.
- **`network/`**: Implements the main network and CoMP configurations for the study.
- **`propagation/`**: Scripts for simulating propagation models used in the aerial RIS-CoMP-NOMA setup.
- **`stats/`**: Statistical analysis utilities for calculating spectral efficiency, SINR, and outage probability.
- **`utils/`**: Utility functions for data processing, encoding, and decoding operations.
- **`main.py`** and **`main2.py`**: Primary scripts for running the simulations, which can be configured for specific setups (RIS elements, NOMA configurations, etc.).
- **Data Files**:
  - `RIS_NOMA.csv`: Contains raw data related to RIS-NOMA configurations.
  - `RIS_NOMA_efficiency.csv`: Data on efficiency metrics under RIS-NOMA.
  - `RIS_NOMA_outage.csv`: Outage probability data for various configurations.
  - `RIS_NOMA_sumrate.csv`: Sum rate data for network configurations.
- **Images**:
  - `noma_ris_rate.png`: Illustrative figure of the achieved data rate vs. the number of RIS elements.

### Key Components

1. **Aerial RIS Configuration**:
   - Uses UAV-mounted RIS to dynamically cover urban areas, enhancing network connectivity and providing a virtual line-of-sight (VLOS) link for blocked urban channels.

2. **CoMP and NOMA Integration**:
   - Uses CoMP to reduce inter-cell interference in multi-cell networks by coordinating among base stations (BS).
   - NOMA provides spectral efficiency improvements through power-domain multiplexing, allowing multiple users to share the same frequency band simultaneously.

3. **Machine Learning-Driven Compressed Communication**:
   - The repository employs autoencoders (CNN, CNN+Attention, RNN, Transformer) for compressing Quantized Phase Shift (QPS) feedback from the receiver to RIS, reducing feedback overhead.

### Simulation Results

The simulations explore:
- **Spectral Efficiency and Energy Efficiency**: Analyzed as functions of transmitted power and RIS configurations.
- **Outage Probability and Average Rate**: Evaluates the impact of RIS and CoMP-NOMA on reducing outage events and improving user data rates.
- **Impact of RIS Elements**: Shows network performance increases with more RIS elements, enhancing signal strength through optimal reflection.

### Running the Simulations

1. Clone the repository and install dependencies from `requirements.txt`.
2. Configure parameters in `main.py` or `main2.py`:
   - Adjust RIS element count, NOMA configurations, or CoMP settings.
3. Run the simulations:
   ```bash
   python main.py
   ```
4. Results will be generated and saved in CSV format or plotted as PNG figures (e.g., `noma_ris_rate.png`).

### Requirements

- Python 3.8+
- Libraries: `torch`, `numpy`, `matplotlib`, `pandas`, `scipy`

### License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 

---

