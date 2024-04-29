NAME?=semantic-editing
GPUS?=1
DATA_ROOT?=/s3/aidar/semantic-editing
NOTEBOOKS?=/s3/aidar/notebooks/semantic-editing
CHECKPOINTS?=/s3/aidar/checkpoints/semantic-editing

.PHONY: build run stop

build:
	docker build -t $(NAME) .

run:
	docker run --rm -it --runtime=nvidia \
	-e NVIDIA_VISIBLE_DEVICES=$(GPUS) \
	--ipc=host \
	--net=host \
	-v $(PWD):/workspace \
	-v $(DATA_ROOT):/workspace/data \
	-v $(NOTEBOOKS):/workspace/notebooks \
	-v $(CHECKPOINTS):/workspace/saved \
	--name=$(NAME) \
	$(NAME) \
	bash
