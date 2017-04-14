img = 
db = alldb

evaluate:
	python mlp.py --dataset $(db)

classify:
ifeq ($(img),)
	@echo usage: make classify img=x, where inputs/x.jpg
else
	python mlp.py --dataset $(db) -c inputs/$(img).jpg
endif
