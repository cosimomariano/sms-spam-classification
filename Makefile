.PHONY: train serve test generate-openapi
train:
	python -m scripts.train_pipeline
serve:
	uvicorn app.main:app --host 0.0.0.0 --port 8000
test:
	pytest
generate-openapi:
	python -m scripts.generate_openapi_models
