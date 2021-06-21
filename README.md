# PHM Data Challenge 2021

The repository is a partial attempt to solve the PHM DataChallenge 2021 [https://phm-europe.org/data-challenge](https://phm-europe.org/data-challenge).

The initial idea to solve the challenge consists in the implementation of the Auto-Associative Kernel Regression (AAKR) technique and subsequently in the implementation of a modified AAKR, see Baraldi et al. [https://doi.org/10.1016/j.ymssp.2014.09.013](https://doi.org/10.1016/j.ymssp.2014.09.013).

## Get started
The project requires Python >= 3.8. Create a vitual environment and install project requirements running the followign commands in a bash shell:

	python3 -m venv <name_of_your_virtual_environment>
	source <name_of_your_virtual_environment>/bin/activate
	pip install -r requirements.txt

Then, run the web-app to visualize the observed and reconstructed signals using the following command:

	python3 wsgy.py

Finally, open a web browser and navigate to `localhost:5010`.
