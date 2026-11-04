
Execute worker (from project container)

    1. First start minikube
        minikube start
    2. check fraud-detection-postgresql pod is up
        kubectl get pods -A
    3. port-forward to access postgresql servive from outside minikube cluster
        kubectl port-forward svc/fraud-postgres-postgresql 5432:5432
    4. Execute worker
        uv run fraud-detection-processor