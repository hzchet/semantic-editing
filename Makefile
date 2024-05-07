NAME?=semantic-editing
GPUS?=all
DATA_ROOT?=/s3/aidar/semantic-editing
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
	-v $(CHECKPOINTS):/workspace/saved \
	--name=$(NAME) \
	$(NAME) \
	bash
