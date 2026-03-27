#!/usr/bin/env bash
set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Check Docker ──────────────────────────────────────────────
check_docker() {
    if ! command -v docker &>/dev/null; then
        error "Docker is not installed. Please install Docker Desktop from https://www.docker.com/products/docker-desktop/"
    fi
    if ! docker info &>/dev/null; then
        error "Docker daemon is not running. Please start Docker Desktop."
    fi
    info "Docker is running: $(docker --version)"
}

# ── Check/Install kubectl ────────────────────────────────────
check_kubectl() {
    if ! command -v kubectl &>/dev/null; then
        info "Installing kubectl..."
        if command -v choco &>/dev/null; then
            choco install kubernetes-cli -y
        elif command -v winget &>/dev/null; then
            winget install -e --id Kubernetes.kubectl
        else
            curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/windows/amd64/kubectl.exe"
            mv kubectl.exe /usr/local/bin/kubectl
        fi
    fi
    info "kubectl: $(kubectl version --client --short 2>/dev/null || kubectl version --client)"
}

# ── Check/Install k3d ────────────────────────────────────────
check_k3d() {
    if ! command -v k3d &>/dev/null; then
        info "Installing k3d..."
        if command -v choco &>/dev/null; then
            choco install k3d -y
        else
            curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
        fi
    fi
    info "k3d: $(k3d version)"
}

# ── Check/Install Helm ───────────────────────────────────────
check_helm() {
    if ! command -v helm &>/dev/null; then
        info "Installing Helm..."
        if command -v choco &>/dev/null; then
            choco install kubernetes-helm -y
        else
            curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
        fi
    fi
    info "Helm: $(helm version --short)"
}

# ── Check Go ─────────────────────────────────────────────────
check_go() {
    if ! command -v go &>/dev/null; then
        error "Go is not installed. Please install Go 1.22+ from https://go.dev/dl/"
    fi
    GO_VERSION=$(go version | awk '{print $3}')
    info "Go: $GO_VERSION"
}

# ── Check Python ─────────────────────────────────────────────
check_python() {
    PYTHON_CMD=""
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        error "Python is not installed. Please install Python 3.10+ from https://python.org"
    fi
    info "Python: $($PYTHON_CMD --version)"

    if [ ! -d "$SCRIPT_DIR/.venv" ]; then
        info "Creating Python virtual environment..."
        $PYTHON_CMD -m venv "$SCRIPT_DIR/.venv"
    fi
    info "Python venv ready at .venv/"
}

# ── Check Node.js ────────────────────────────────────────────
check_node() {
    if ! command -v node &>/dev/null; then
        error "Node.js is not installed. Please install Node.js 20+ from https://nodejs.org"
    fi
    info "Node.js: $(node --version)"
}

# ── Create k3d cluster ───────────────────────────────────────
create_cluster() {
    if k3d cluster list 2>/dev/null | grep -q "skam-chaos"; then
        warn "Cluster 'skam-chaos' already exists. Use 'make cluster-down' to delete it first."
        return 0
    fi
    info "Creating k3d cluster 'skam-chaos'..."
    k3d cluster create --config "$SCRIPT_DIR/k8s/cluster/k3d-config.yaml"
    info "Waiting for cluster to be ready..."
    kubectl wait --for=condition=Ready nodes --all --timeout=120s
    info "Cluster is ready!"
}

# ── Install Python dependencies ──────────────────────────────
install_python_deps() {
    info "Installing Python dependencies..."
    source "$SCRIPT_DIR/.venv/bin/activate" 2>/dev/null || source "$SCRIPT_DIR/.venv/Scripts/activate"
    pip install --upgrade pip -q
    pip install -r "$SCRIPT_DIR/requirements.txt" -q
    info "Python dependencies installed."
}

# ── Main ─────────────────────────────────────────────────────
main() {
    info "=========================================="
    info " SKAM - Chaos Engineering Platform Setup"
    info "=========================================="

    check_docker
    check_kubectl
    check_k3d
    check_helm
    check_go
    check_python
    check_node

    echo ""
    info "All prerequisites satisfied!"
    echo ""

    read -p "Create k3d cluster now? [Y/n] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        create_cluster
        install_python_deps
        info "Setup complete! Run 'make build && make deploy' to get started."
    else
        info "Skipped cluster creation. Run 'make cluster-up' when ready."
    fi
}

main "$@"
