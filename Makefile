requirements.txt: Pipfile.lock
	pipenv lock --dev -r > requirements.txt

.PHONY: test
test: requirements.txt
	pipenv run tox

.PHONY: coverage
coverage:
	pipenv run nose2 --with-coverage

.PHONY: docs
docs:
	pipenv run tox -e docs
