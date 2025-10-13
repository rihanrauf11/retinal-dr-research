#!/bin/bash

# build_and_push.sh
# Automated Docker image build and push script
#
# Usage:
#   bash scripts/build_and_push.sh [OPTIONS]
#
# Options:
#   --username <name>    Docker Hub username (required)
#   --tag <tag>          Image tag (default: latest)
#   --no-cache           Build without cache
#   --push               Push to Docker Hub after building
#   --test               Run tests before building
#
# Examples:
#   # Build only
#   bash scripts/build_and_push.sh --username myusername
#
#   # Build and push
#   bash scripts/build_and_push.sh --username myusername --push
#
#   # Build with specific tag
#   bash scripts/build_and_push.sh --username myusername --tag v1.0.0 --push
#
#   # Build without cache
#   bash scripts/build_and_push.sh --username myusername --no-cache

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Default values
DOCKER_USERNAME=""
IMAGE_TAG="latest"
USE_CACHE=true
SHOULD_PUSH=false
RUN_TESTS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --username)
            DOCKER_USERNAME="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --no-cache)
            USE_CACHE=false
            shift
            ;;
        --push)
            SHOULD_PUSH=true
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo ""
            echo "Usage: bash scripts/build_and_push.sh --username <name> [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --username <name>    Docker Hub username (required)"
            echo "  --tag <tag>          Image tag (default: latest)"
            echo "  --no-cache           Build without cache"
            echo "  --push               Push to Docker Hub after building"
            echo "  --test               Run tests before building"
            exit 1
            ;;
    esac
done

# Check if username is provided
if [ -z "$DOCKER_USERNAME" ]; then
    print_error "Docker Hub username is required"
    echo ""
    echo "Usage: bash scripts/build_and_push.sh --username <your-dockerhub-username>"
    exit 1
fi

# Set image name
IMAGE_NAME="$DOCKER_USERNAME/dr-retfound-lora"
FULL_IMAGE="$IMAGE_NAME:$IMAGE_TAG"

echo "=========================================="
echo "  Docker Build & Push Script"
echo "=========================================="
echo ""
echo "Configuration:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Docker Username: $DOCKER_USERNAME"
echo "Image Name:      $IMAGE_NAME"
echo "Tag:             $IMAGE_TAG"
echo "Full Image:      $FULL_IMAGE"
echo "Use Cache:       $USE_CACHE"
echo "Push:            $SHOULD_PUSH"
echo "Run Tests:       $RUN_TESTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    print_info "Install from: https://docs.docker.com/get-docker/"
    exit 1
fi

print_success "Docker is installed"

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running"
    print_info "Start Docker Desktop or Docker daemon"
    exit 1
fi

print_success "Docker daemon is running"

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    print_error "Dockerfile not found in current directory"
    exit 1
fi

print_success "Dockerfile found"

# Run tests if requested
if [ "$RUN_TESTS" = true ]; then
    echo ""
    print_info "Running tests..."

    if command -v pytest &> /dev/null; then
        pytest tests/ -v --tb=short
        if [ $? -eq 0 ]; then
            print_success "All tests passed"
        else
            print_error "Tests failed"
            exit 1
        fi
    else
        print_warning "pytest not installed, skipping tests"
    fi
fi

# Display Docker image size estimate
echo ""
print_info "Estimating image size..."
echo "  Base image (pytorch/pytorch:2.1.0-cuda12.1): ~7GB"
echo "  Dependencies: ~1GB"
echo "  Project files: ~50MB"
echo "  Estimated total: ~8GB"
echo ""

# Build Docker image
echo "=========================================="
print_info "Building Docker image..."
echo "=========================================="
echo ""

BUILD_CMD="docker build -t $FULL_IMAGE"

# Add no-cache flag if specified
if [ "$USE_CACHE" = false ]; then
    BUILD_CMD="$BUILD_CMD --no-cache"
    print_warning "Building without cache (slower but ensures fresh build)"
fi

BUILD_CMD="$BUILD_CMD ."

# Display build command
echo "Build command:"
echo "$BUILD_CMD"
echo ""

# Execute build
START_TIME=$(date +%s)

$BUILD_CMD

BUILD_STATUS=$?
END_TIME=$(date +%s)
BUILD_TIME=$((END_TIME - START_TIME))

if [ $BUILD_STATUS -eq 0 ]; then
    echo ""
    print_success "Docker image built successfully in ${BUILD_TIME}s"

    # Display image information
    echo ""
    echo "Image information:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    docker images $FULL_IMAGE --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo ""
    print_error "Docker image build failed"
    exit 1
fi

# Tag as latest if not already
if [ "$IMAGE_TAG" != "latest" ]; then
    echo ""
    print_info "Tagging as latest..."
    docker tag $FULL_IMAGE $IMAGE_NAME:latest
    print_success "Tagged as $IMAGE_NAME:latest"
fi

# Push to Docker Hub if requested
if [ "$SHOULD_PUSH" = true ]; then
    echo ""
    echo "=========================================="
    print_info "Pushing to Docker Hub..."
    echo "=========================================="
    echo ""

    # Check if logged in to Docker Hub
    if ! docker info 2>/dev/null | grep -q "Username: $DOCKER_USERNAME"; then
        print_warning "Not logged in to Docker Hub"
        print_info "Logging in..."
        docker login
    fi

    print_success "Logged in to Docker Hub"

    # Push image
    echo ""
    print_info "Pushing $FULL_IMAGE..."
    docker push $FULL_IMAGE

    if [ $? -eq 0 ]; then
        print_success "Successfully pushed $FULL_IMAGE"

        # Push latest tag if different
        if [ "$IMAGE_TAG" != "latest" ]; then
            print_info "Pushing $IMAGE_NAME:latest..."
            docker push $IMAGE_NAME:latest

            if [ $? -eq 0 ]; then
                print_success "Successfully pushed $IMAGE_NAME:latest"
            fi
        fi
    else
        print_error "Failed to push image"
        exit 1
    fi
fi

# Display summary
echo ""
echo "=========================================="
print_success "Build Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Image:       $FULL_IMAGE"
echo "Build Time:  ${BUILD_TIME}s"

if [ "$SHOULD_PUSH" = true ]; then
    echo "Status:      Built and pushed to Docker Hub"
else
    echo "Status:      Built locally (not pushed)"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Display next steps
echo "Next steps:"
echo ""

if [ "$SHOULD_PUSH" = false ]; then
    echo "1. Push to Docker Hub:"
    echo "   docker push $FULL_IMAGE"
    echo ""
fi

echo "2. Test image locally:"
echo "   docker run --gpus all -it $FULL_IMAGE"
echo ""

echo "3. Use on Vast.ai:"
echo "   a. Create instance with custom Docker image"
echo "   b. Specify: $FULL_IMAGE"
echo "   c. Mount volumes:"
echo "      /data    → Your datasets"
echo "      /models  → RETFound weights"
echo "      /results → Training outputs"
echo ""

echo "4. Run training on Vast.ai:"
echo "   bash scripts/vast_setup.sh"
echo "   bash scripts/vast_train.sh"
echo ""

print_info "Docker image ready for deployment!"
