test:
	test.sh

conda-build:
	conda env create -f environment.yml

oasis-install:
    pip install git+https://github.com/j-friedrich/OASIS.git