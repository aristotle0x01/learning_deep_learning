env setup use uv instead of conda

```shell
uv python install 3.11
cd ml-zoomcamp
uv python pin 3.11
uv add numpy pandas scikit-learn matplotlib seaborn jupyter

# run 1
uv run jupyter notebook
# or 2
source .venv/bin/activate
jupyter notebook
```
