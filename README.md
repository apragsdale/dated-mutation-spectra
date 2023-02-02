## Requirements

We need to install with Python versions >=3.7,<3.11.

Install requirements as 

```
pip install -r requirements.txt
```

## Getting data

TODO: write a script to get, organize, and format data. Or
at least point to where it all can be downloaded from. They
are all public and well-known resources.

All data is stored in the `data` directory, including

- Reference genomes
- GEVA Atlas allele ages
- tsdate-inferred trees
- Relate-inferred allele ages
- Thousand Genomes Project VCFs and samples file

## Running analyses

### Computing and visualizing the spectrum history
To parse data, we run

```
python process_{dataset}.py [optional] -c {chrom}
```

which creates a set of dataframes with mutation types in each chromosome.
This can take a while...

These can be binned into time-bins using

```
python bin_variant_ages.py -d {dataset} [optional] -m {max_age} -s -f {max_frequency}
```

where the `-s` flag is used if we want to include singletons. If singletons
should be excluded, omit the `-s` flag.

These can then be plotted using `plot_spectra_history.py`, which takes
the same arguments as `bin_variant_ages.py`.

### Fitting a model for age- and sex-dependent mutation rates

From the Iceland trio data (Jonnson et al, 2017), fit a linear model for
the mutation spectrum that depends on the ages of each parent, using

```
python xyz.py
```

The output is stored as a XXX, which can be used as YYY.

### Inferring generation time history

Using the mutation spectra binned by allele age, and the regression of
de novo mutation profiles on parental age and sex, fit the sex-specific
generation time history, as

```
python xyz.py -d {dataset}
```
