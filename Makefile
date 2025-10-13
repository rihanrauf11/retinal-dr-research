# Makefile for Diabetic Retinopathy Classification Project
# Provides convenient commands for Docker operations, testing, and deployment
#
# Usage:
#   make help              Show this help message
#   make docker-build      Build Docker image
#   make docker-push       Push Docker image to Docker Hub
#   make docker-run        Run Docker container locally
#   make test              Run pytest tests
#   make clean             Clean up temporary files

.PHONY: help docker-build docker-push docker-run docker-shell test clean format lint

# Configuration
DOCKER_USERNAME ?= your-dockerhub-username
IMAGE_NAME = $(DOCKER_USERNAME)/dr-retfound-lora
IMAGE_TAG ?= latest
FULL_IMAGE = $(IMAGE_NAME):$(IMAGE_TAG)

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo ""
	@echo "$(BLUE)Diabetic Retinopathy Classification - Makefile Commands$(NC)"
	@echo "========================================================"
	@echo ""
	@echo "$(GREEN)Docker Commands:$(NC)"
	@grep -E '^docker-[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Development Commands:$(NC)"
	@grep -E '^(test|clean|format|lint):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(GREEN)Configuration:$(NC)"
	@echo "  DOCKER_USERNAME = $(DOCKER_USERNAME)"
	@echo "  IMAGE_NAME      = $(IMAGE_NAME)"
	@echo "  IMAGE_TAG       = $(IMAGE_TAG)"
	@echo ""
	@echo "$(BLUE)Examples:$(NC)"
	@echo "  make docker-build DOCKER_USERNAME=myusername"
	@echo "  make docker-push DOCKER_USERNAME=myusername IMAGE_TAG=v1.0.0"
	@echo "  make test"
	@echo ""

docker-build: ## Build Docker image
	@echo "$(GREEN)Building Docker image: $(FULL_IMAGE)$(NC)"
	@if [ "$(DOCKER_USERNAME)" = "your-dockerhub-username" ]; then \
		echo "$(RED)Error: Please set DOCKER_USERNAME$(NC)"; \
		echo "Example: make docker-build DOCKER_USERNAME=myusername"; \
		exit 1; \
	fi
	docker build -t $(FULL_IMAGE) .
	@if [ "$(IMAGE_TAG)" != "latest" ]; then \
		docker tag $(FULL_IMAGE) $(IMAGE_NAME):latest; \
		echo "$(GREEN)Also tagged as: $(IMAGE_NAME):latest$(NC)"; \
	fi
	@echo "$(GREEN)✓ Build complete!$(NC)"
	@docker images $(IMAGE_NAME)

docker-build-no-cache: ## Build Docker image without cache
	@echo "$(GREEN)Building Docker image without cache: $(FULL_IMAGE)$(NC)"
	@if [ "$(DOCKER_USERNAME)" = "your-dockerhub-username" ]; then \
		echo "$(RED)Error: Please set DOCKER_USERNAME$(NC)"; \
		exit 1; \
	fi
	docker build --no-cache -t $(FULL_IMAGE) .
	@echo "$(GREEN)✓ Build complete!$(NC)"

docker-push: ## Push Docker image to Docker Hub
	@echo "$(GREEN)Pushing Docker image: $(FULL_IMAGE)$(NC)"
	@if [ "$(DOCKER_USERNAME)" = "your-dockerhub-username" ]; then \
		echo "$(RED)Error: Please set DOCKER_USERNAME$(NC)"; \
		exit 1; \
	fi
	docker push $(FULL_IMAGE)
	@if [ "$(IMAGE_TAG)" != "latest" ]; then \
		docker push $(IMAGE_NAME):latest; \
	fi
	@echo "$(GREEN)✓ Push complete!$(NC)"

docker-build-and-push: docker-build docker-push ## Build and push Docker image

docker-run: ## Run Docker container locally (requires GPU)
	@echo "$(GREEN)Running Docker container: $(FULL_IMAGE)$(NC)"
	@if [ "$(DOCKER_USERNAME)" = "your-dockerhub-username" ]; then \
		echo "$(RED)Error: Please set DOCKER_USERNAME$(NC)"; \
		exit 1; \
	fi
	docker run --gpus all \
		-v $(PWD)/data:/data \
		-v $(PWD)/models:/models \
		-v $(PWD)/results:/results \
		-it $(FULL_IMAGE) /bin/bash

docker-shell: ## Open shell in Docker container
	@echo "$(GREEN)Opening shell in: $(FULL_IMAGE)$(NC)"
	@if [ "$(DOCKER_USERNAME)" = "your-dockerhub-username" ]; then \
		echo "$(RED)Error: Please set DOCKER_USERNAME$(NC)"; \
		exit 1; \
	fi
	docker run --gpus all \
		-v $(PWD)/data:/data \
		-v $(PWD)/models:/models \
		-v $(PWD)/results:/results \
		-it $(FULL_IMAGE) /bin/bash

docker-test-local: ## Test Docker image locally (no GPU required)
	@echo "$(GREEN)Testing Docker image locally$(NC)"
	@if [ "$(DOCKER_USERNAME)" = "your-dockerhub-username" ]; then \
		echo "$(RED)Error: Please set DOCKER_USERNAME$(NC)"; \
		exit 1; \
	fi
	docker run --rm $(FULL_IMAGE) python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	docker run --rm $(FULL_IMAGE) python -c "import timm, transformers, peft; print('✓ All dependencies OK')"
	@echo "$(GREEN)✓ Docker image test passed!$(NC)"

test: ## Run pytest tests
	@echo "$(GREEN)Running tests...$(NC)"
	pytest tests/ -v
	@echo "$(GREEN)✓ Tests complete!$(NC)"

test-coverage: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	pytest tests/ -v --cov=scripts --cov-report=html
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(NC)"

clean: ## Clean up temporary files and caches
	@echo "$(GREEN)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	rm -rf .coverage htmlcov/ .tox/
	@echo "$(GREEN)✓ Cleanup complete!$(NC)"

format: ## Format code with black
	@echo "$(GREEN)Formatting code with black...$(NC)"
	black scripts/ tests/
	@echo "$(GREEN)✓ Formatting complete!$(NC)"

lint: ## Lint code with flake8
	@echo "$(GREEN)Linting code with flake8...$(NC)"
	flake8 scripts/ tests/ --max-line-length=100
	@echo "$(GREEN)✓ Linting complete!$(NC)"

check: format lint test ## Run format, lint, and test

# Quick commands for Vast.ai workflow
vast-build: ## Build Docker image for Vast.ai (alias for docker-build)
	@$(MAKE) docker-build

vast-push: ## Push Docker image for Vast.ai (alias for docker-push)
	@$(MAKE) docker-push

vast-ready: docker-build docker-push ## Build and push image ready for Vast.ai
	@echo ""
	@echo "$(GREEN)========================================$(NC)"
	@echo "$(GREEN)✓ Docker image ready for Vast.ai!$(NC)"
	@echo "$(GREEN)========================================$(NC)"
	@echo ""
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "1. Go to Vast.ai and create an instance"
	@echo "2. Use Docker image: $(FULL_IMAGE)"
	@echo "3. Mount volumes:"
	@echo "   /data    → Your datasets"
	@echo "   /models  → RETFound weights"
	@echo "   /results → Training outputs"
	@echo "4. SSH into instance and run:"
	@echo "   bash scripts/vast_setup.sh"
	@echo "   bash scripts/vast_train.sh"
	@echo ""

# Info commands
info: ## Display project information
	@echo ""
	@echo "$(BLUE)Project Information$(NC)"
	@echo "===================="
	@echo "Docker Image: $(FULL_IMAGE)"
	@echo "Python:       $(shell python --version 2>&1)"
	@echo "PyTorch:      $(shell python -c 'import torch; print(torch.__version__)' 2>&1)"
	@echo "CUDA:         $(shell python -c 'import torch; print(torch.cuda.is_available())' 2>&1)"
	@echo ""

docker-info: ## Display Docker image information
	@echo ""
	@echo "$(BLUE)Docker Images$(NC)"
	@echo "=============="
	@docker images $(IMAGE_NAME) 2>/dev/null || echo "No images found"
	@echo ""

# Installation helpers
install-dev: ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(NC)"
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8
	@echo "$(GREEN)✓ Installation complete!$(NC)"

verify-install: ## Verify installation
	@echo "$(GREEN)Verifying installation...$(NC)"
	@python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
	@python -c "import torchvision; print(f'✓ torchvision {torchvision.__version__}')"
	@python -c "import timm; print(f'✓ timm {timm.__version__}')"
	@python -c "import transformers; print(f'✓ transformers {transformers.__version__}')"
	@python -c "import peft; print(f'✓ peft {peft.__version__}')"
	@python -c "import albumentations; print(f'✓ albumentations {albumentations.__version__}')"
	@echo "$(GREEN)✓ All dependencies installed correctly!$(NC)"
