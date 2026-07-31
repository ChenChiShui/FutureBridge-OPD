# Runtime provenance

The released overlay was recovered from a TCOD workspace based on commit
`17a8af222017fb78a44a50e5711bfa08543de36b`.

For a clean public installation, use `requirements.txt`, which follows the
dependency declarations of that pinned TCOD revision, including veRL 0.7.0 and
vLLM 0.10.2--0.14.1. The table below records the older server environment from
which the workflow files were recovered; it is provenance rather than the
recommended fresh-install lock.

The recorded server environment used Python 3.10--3.12-compatible TCOD code
with the following core package versions:

| Package | Version |
| --- | --- |
| PyTorch | 2.6.0 |
| Transformers | 4.51.3 |
| vLLM | 0.8.5.post1 |
| Ray | 2.54.1 |
| veRL | 0.6.1 |
| OmegaConf | 2.3.0 |
| ALFWorld | 0.4.2 |
| TextWorld | 1.6.2 |
| ScienceWorld | 1.2.2 |
| Gym | 0.24.0 |
| Weights & Biases | 0.28.1 |

WebShop is installed from its source tree rather than a versioned Python
package. Its data, search index, and source checkout must be prepared before
running the WebShop configurations. ScienceWorld task records embed the local
path to `scienceworld.jar`; regenerate the task data after installing the JAR
on a new machine.
