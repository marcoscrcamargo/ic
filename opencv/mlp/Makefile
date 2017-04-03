img = 

evaluate:
	python mlp.py --dataset DBIM

classify:
ifeq ($(img),)
	@echo usage: make classify img=x, where inputs/x.jpg
else
	python mlp.py --dataset DBIM -c inputs/$(img).jpg
endif
