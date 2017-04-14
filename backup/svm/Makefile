img = 
db = alldb

evaluate:
	python svm.py --dataset $(db)

classify:
ifeq ($(img),)
	@echo usage: make classify img=x, where inputs/x.jpg
else
	python svm.py --dataset $(db) -c inputs/$(img).jpg
endif
