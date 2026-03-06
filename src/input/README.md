# input

This module holds the following files: 

`params.py`:
`preprocess.py`
`features.py`
`input.py`: The main file of this module. 

## Usage 

To preprocess 

```python
import polars as pl

from input.input import get_partition

df = get_partition()
```
