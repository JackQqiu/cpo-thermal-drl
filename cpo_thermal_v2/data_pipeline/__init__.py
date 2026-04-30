"""
cpo_thermal_v2.data_pipeline — offline preprocessing scripts
============================================================

The two scripts here are typically invoked as modules:

    python -m cpo_thermal_v2.data_pipeline.compute_dag_features \
        --input  data_pipeline/process/alibaba_dags.json \
        --output data_pipeline/process/alibaba_dags_v2.json

    python -m cpo_thermal_v2.data_pipeline.generate_matrices \
        --output-root data/thermal_matrics/

They can also be imported and called programmatically::

    >>> from cpo_thermal_v2.data_pipeline import enrich_file, generate_for_size
"""
from .compute_dag_features import enrich_dag, enrich_file
from .generate_matrices    import (
    generate_for_size, generate_batch, build_state_space_matrices,
    DEFAULT_OE_LIST,
)

__all__ = [
    "enrich_dag", "enrich_file",
    "generate_for_size", "generate_batch", "build_state_space_matrices",
    "DEFAULT_OE_LIST",
]
