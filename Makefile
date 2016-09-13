clean:
	find . -name "*~" | xargs rm -f
	rm -rf build/
	rm -rf bin/
