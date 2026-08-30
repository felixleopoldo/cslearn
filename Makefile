.PHONY: distr

distr:
	rm -rf dist build
	python -m build
	twine upload dist/*
