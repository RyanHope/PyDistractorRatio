all: extension
	cp build/lib.*/*.so .

extension:
	python setup.py build
	
clean:
	rm -rf *.so
	rm -rf build