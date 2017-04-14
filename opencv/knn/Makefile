img = 
db = alldb

evaluate:
	python knn.py --dataset $(db)

classify:
ifeq ($(img),)
	@echo usage: make classify img=x, where inputs/x.jpg
else
	python knn.py --dataset $(db) -c inputs/$(img).jpg
endif
