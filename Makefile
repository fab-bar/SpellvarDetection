REBUILD_FLAG =

requirements.txt: Pipfile.lock
	$(eval REBUILD_FLAG := --recreate)
	pipenv lock --dev -r > requirements.txt


.PHONY: test
test: requirements.txt
	pipenv run tox $(REBUILD_FLAG)

.PHONY: coverage
coverage:
	pipenv run nose2 --with-coverage

.PHONY: docs
docs:
	pipenv run $(MAKE) -C docs html
