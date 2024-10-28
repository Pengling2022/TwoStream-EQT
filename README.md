# TS-EQT

## Environment Requirements

This project requires the use of the `conda` package management tool for environment management. We recommend using Python 3.9.

## Installation Steps

Please follow the steps below to set up your development environment:

1.  **Create a New Conda Environment** Use the following command to create a new Conda environment named `ts-eqt`, specifying Python version 3.9:

        conda create -n ts-eqt python=3.9

2.  **Activate the Environment** Activate the environment you just created:

        conda activate ts-eqt

3.  **Install Dependencies** Install the required packages based on the `requirements.txt` file in the project:

        pip install -r requirements.txt

4.  **Install the Local seisbench Package**

        pip install .

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please submit an issue or pull request.

## License

This project is licensed under the GNU General Public License v3.0.

## References

Peng L, Li L, Mousavi S M, et al. TwoStream-EQT: A microseismic phase picking model combining time and frequency domain inputs. submitted manuscript, 2024.

Mousavi S M, Ellsworth W L, Zhu W, et al. Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking. Nature Communications, 2020, 11(1): 3952.

Zhu W, Beroza G C. PhaseNet: a deep-neural-network-based seismic arrival-time picking method. Geophysical Journal International, 2019, 216(1): 261-273.

Zhu W, McBrearty I W, Mousavi S M, et al. Earthquake phase association using a Bayesian Gaussian mixture model. Journal of Geophysical Research: Solid Earth, 2022, 127(5): e2021JB023249.

Woollam J, Münchmeyer J, Tilmann F, et al. SeisBench—A toolbox for machine learning in seismology. Seismological Research Letters, 2022, 93(3): 1695-1709.

Zhao M, Xiao Z, Chen S, et al. DiTing: A large-scale Chinese seismic benchmark dataset for artificial intelligence in seismology. Earthquake Science, 2022, 35: 1-11.

Mousavi S M, Sheng Y, Zhu W, et al. STanford EArthquake Dataset (STEAD): A global data set of seismic signals for AI. IEEE Access, 2019, 7: 179464-179476
