img = 

evaluate:
	python knn.py --dataset DBIM

classify:
ifeq ($(img),)
	@echo usage: make classify img=x, where inputs/x.jpg
else
	python knn.py --dataset DBIM -c inputs/$(img).jpg
endif
