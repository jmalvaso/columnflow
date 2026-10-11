# coding: utf-8
"""Select BDT mass categories without changing event inputs."""

import re
import law

_BDT_MASS_RE = re.compile(r"__bdt_.+_M([0-9]+)$")


class HistogramCategoriesMixin:
    histogram_bdt_masses = law.CSVParameter(
        default=(),
        description="BDT masses retained for ordinary variables: 100, 100,200, or all; "
        "empty keeps the parent-region layout",
    )

    @classmethod
    def resolve_param_values_pre_init(cls, params):
        params = super().resolve_param_values_pre_init(params)
        values = tuple(str(value).strip() for value in params.get("histogram_bdt_masses", ()))
        if not values:
            return params
        if values != ("all",) and not all(re.fullmatch(r"[1-9][0-9]*", value) for value in values):
            raise ValueError("--histogram-bdt-masses requires positive mass points or 'all'")

        container = cls._get_config_container(params)
        if container is None:
            return params
        configs = container if isinstance(container, (list, tuple)) else [container]
        grids = [
            {int(match.group(1)) for category in config.get_leaf_categories()
             if (match := _BDT_MASS_RE.search(category.name))}
            for config in configs
        ]
        if values == ("all",):
            if not grids[0] or any(grid != grids[0] for grid in grids[1:]):
                raise ValueError("'all' requires the same nonempty BDT mass grid in every config")
            masses = grids[0]
        else:
            masses = set(map(int, values))
        for config, grid in zip(configs, grids):
            if masses - grid:
                raise ValueError(f"BDT masses {sorted(masses - grid)} are unavailable in {config.name}")
        params["histogram_bdt_masses"] = tuple(str(mass) for mass in sorted(masses))
        return params

    @classmethod
    def req_params(cls, inst, **kwargs):
        kwargs["_prefer_cli"] = law.util.make_set(kwargs.get("_prefer_cli", [])) | {"histogram_bdt_masses"}
        return super().req_params(inst, **kwargs)

    @property
    def histogram_bdt_masses_repr(self):
        masses = tuple(self.histogram_bdt_masses)
        label = "M" + "_".join(masses)
        if len(label) > 60:
            label = f"{len(masses)}m_{law.util.create_hash(masses)}"
        return label

    def store_parts(self):
        parts = super().store_parts()
        if self.histogram_bdt_masses:
            parts.insert_before("version", "histogram_bdt_masses", f"bdtcats__{self.histogram_bdt_masses_repr}")
        return parts

    def control_output_postfix(self):
        postfix = super().control_output_postfix()
        if self.histogram_bdt_masses:
            postfix += f"__bdtcats_{self.histogram_bdt_masses_repr}"
        return postfix