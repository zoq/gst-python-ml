# Building PyPI Package

## Setup
1. Generate token on PyPI and copy to `.pypirc`

```
[pypi]
  username = __token__
  password = $TOKEN
```

2. Install build dependencies

```
pip install setuptools wheel twine
pip install --upgrade build
```
## Build and Upoad

```
python -m build
twine upload dist/*
```
