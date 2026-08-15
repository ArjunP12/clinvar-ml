.PHONY: install run clean data

install:
	pip install -r requirements.txt

run:
	python main.py

clean:
	rm -rf results/*.png results/*.csv results/ablations
	rm -f data/annotations_cache.csv

data:
	python scripts/download_clinvar.py
