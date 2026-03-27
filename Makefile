.PHONY: setup cluster-up cluster-down build import-images deploy deploy-infra deploy-services deploy-platform undeploy load-test chaos-demo dashboard clean

REGISTRY := localhost:5111
NAMESPACE := default
PLATFORM_NS := skam-platform
SERVICES := api-gateway user-service product-service order-service payment-service notification-service
PLATFORM_SERVICES := chaos-engine anomaly-detector decision-engine

# ── Setup ─────────────────────────────────────────────────────
setup:
	bash setup.sh

# ── Cluster Management ────────────────────────────────────────
cluster-up:
	k3d cluster create --config k8s/cluster/k3d-config.yaml
	kubectl create namespace $(PLATFORM_NS) --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f k8s/rbac/
	$(MAKE) deploy-infra

cluster-down:
	k3d cluster delete skam-chaos

# ── Build Docker Images ──────────────────────────────────────
build: build-services build-platform

build-services:
	@for svc in $(SERVICES); do \
		echo "Building $$svc..."; \
		docker build -t $(REGISTRY)/skam/$$svc:latest services/$$svc/; \
	done

build-platform:
	@for svc in $(PLATFORM_SERVICES); do \
		echo "Building $$svc..."; \
		docker build -t $(REGISTRY)/skam/$$svc:latest platform/$$svc/; \
	done

# ── Push Images to Local Registry ────────────────────────────
push-images:
	@for svc in $(SERVICES) $(PLATFORM_SERVICES); do \
		echo "Pushing $$svc..."; \
		docker push $(REGISTRY)/skam/$$svc:latest; \
	done

# ── Deploy ───────────────────────────────────────────────────
deploy: deploy-infra deploy-services deploy-platform

deploy-infra:
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
	helm repo add grafana https://grafana.github.io/helm-charts 2>/dev/null || true
	helm repo update
	helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
		--namespace monitoring --create-namespace \
		-f k8s/infrastructure/prometheus-values.yaml --wait --timeout 5m
	helm upgrade --install loki grafana/loki-stack \
		--namespace monitoring \
		-f k8s/infrastructure/loki-values.yaml --wait --timeout 5m

deploy-services:
	kubectl apply -f k8s/microservices/postgres-deployment.yaml
	kubectl apply -f k8s/microservices/redis-deployment.yaml
	kubectl wait --for=condition=Ready pod -l app=postgres --timeout=120s
	kubectl wait --for=condition=Ready pod -l app=redis --timeout=120s
	@for svc in $(SERVICES); do \
		kubectl apply -f k8s/microservices/$$svc-deployment.yaml; \
	done

deploy-platform:
	@for svc in $(PLATFORM_SERVICES); do \
		kubectl apply -f k8s/microservices/$$svc-deployment.yaml -n $(PLATFORM_NS); \
	done

undeploy:
	@for svc in $(SERVICES); do \
		kubectl delete -f k8s/microservices/$$svc-deployment.yaml --ignore-not-found; \
	done
	@for svc in $(PLATFORM_SERVICES); do \
		kubectl delete -f k8s/microservices/$$svc-deployment.yaml -n $(PLATFORM_NS) --ignore-not-found; \
	done
	kubectl delete -f k8s/microservices/redis-deployment.yaml --ignore-not-found
	kubectl delete -f k8s/microservices/postgres-deployment.yaml --ignore-not-found

# ── Load Testing ─────────────────────────────────────────────
load-test:
	python scripts/load-generator.py

# ── Chaos Demo ───────────────────────────────────────────────
chaos-demo:
	python scripts/demo-scenarios.py

# ── Dashboard ────────────────────────────────────────────────
dashboard:
	cd dashboard && npm run dev

# ── Grafana Port Forward ─────────────────────────────────────
grafana:
	kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# ── Clean ────────────────────────────────────────────────────
clean:
	$(MAKE) undeploy
	$(MAKE) cluster-down
