.PHONY: notebooks

notebooks: examples/*.ipynb
	jupyter nbconvert --to markdown --execute $^

docs:
	mkdocs build

