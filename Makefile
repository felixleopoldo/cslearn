distr:
	rm -r dist build
	python setup.py bdist_wheel
	twine upload dist/*