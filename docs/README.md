# AIY Maker Kit API docs

This directory holds the source files required to build the API
reference with Sphinx.

If you're looking for the setup and developer docs for `aiymakerkit`,
go to https://g.co/aiy/maker.

## Build the docs

You can build the `aiymakerkit` API reference docs as follows:

```
# To ensure consistent results, use a Python virtual environment:
python3 -m venv ~/.my_venvs/aiydocs
source ~/.my_venvs/aiydocs/bin/activate

# Navigate to this docs/ dir and install the doc dependencies:
cd docs
pip install -r requirements.txt
python3 -m pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0

# Build the docs:
bash makedocs.sh
```

The results are output in `_build/`. The `_build/preview/` files are for local
viewing--just open the `index.html` page. The `_build/web/` files are designed
for publishing on the website.

For more information about the syntax in these RST files, see the
[reStructuredText documentation](http://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html).
