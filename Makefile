.PHONY: build_pg_vector build_rag run_rag

# Build the pg_vector Docker image
build_pg_vector:
	docker build -t pg_vector -f docker_pgvector .

# Build the rag Docker image
build_rag:
	docker build -t rag -f docker-rag .

# Run the rag_app using Docker Compose
run_rag:
	docker compose run rag_app

# Build both images and run rag_app
all: build_pg_vector build_rag run_rag
