CONTAINER_NAME:=coronavirus-covid-19-visualization-prediction
TAG:=$(shell git log -1 --pretty=format:"%H")

.PHONY: build
build: ## Build the docker image.
	docker build \
		$(CACHE_FROM) \
		--build-arg VERSION=$(TAG) \
		-t $(CONTAINER_NAME) .

.PHONY: run
run: ## Run the service using docker-compose.
	docker-compose stop
	docker-compose run --rm coronavirus-visualization bash -c "poetry run jupyter lab --allow-root"

.PHONY: lock-dependencies
lock-dependencies: ## Lock poetry dependencies.
	docker run \
		-v `pwd`:/app \
		-it $(CONTAINER_NAME) poetry lock

.PHONY: lint
lint: ## Run service linting.
	docker run \
		-v $(shell pwd)/src:/app/src \
		-v $(shell pwd)/.pylintrc:/app/.pylintrc \
		$(CONTAINER_NAME) \
		poetry run pylint /app/src

PHONY: debug
debug: test-clean ## Run service unit tests.
	docker run \
		-e COVERAGE_FILE=/app/tests/.coverage \
		-v $(shell pwd):/app \
		-ti \
		$(CONTAINER_NAME) \
		/bin/bash -c "poetry run pytest ${test_dir} -s -v"