#!/bin/bash

sourceFile=Cit-HepTh
SLEEPTIME=1200
NUMBER =
$(foreach num, $(shell seq 0 1 9), $(eval NUMBER += $(num)))
NUMBER1 =
$(foreach num, $(shell seq 0 1 29), $(eval NUMBER += $(num)))


clean:
	rm *.log nohup.out


one:
	for i in $(NUMBER); do \
		nohup python3 -u TriangleCount.py $$i $(sourceFile)$$i & \
	done

run:
	for i in $(NUMBER1); do \
		sleep $(SLEEPTIME); \
		for j in $(NUMBER); do \
			nohup python3 -u TriangleCount.py $$i$$j $(sourceFile) & \
		done \
	done

run_IMDB:
	for i in $(NUMBER1); do \
		sleep $(SLEEPTIME); \
		for j in $(NUMBER); do \
			nohup python3 -u TriangleCount.py $$i$$j $(sourceFile)$$j & \
		done \
	done

show:
	tail -f nohup.out

mean:
	python3 convert.py
	python3 mean.py
	cat re.dat
	cat l2.dat
