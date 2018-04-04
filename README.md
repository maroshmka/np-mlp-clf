# MLP classifier

This is an implementation of raw multilayer perceptron classifier in numpy.  
It has also implementation of simple GridSearch.  

### Usage:

Install dependencies from `requirements.txt`, e.g. to virtualenv.  
Then run script.

```bash
$ python3 main.py [-gs] --data iris|2d
  gs - use grid search for model selection
```

In the end, visualisations of results will be stored in `results/`.
You can see an example of a results in `example_results/`.
