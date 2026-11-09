# coding: utf-8

"""
Base tasks for writing serialized statistical inference models.
"""

from __future__ import annotations

import law
import order as od

from columnflow.tasks.framework.base import Requirements
from columnflow.tasks.framework.mixins import (
    CalibratorClassesMixin, SelectorClassMixin, ReducerClassMixin, ProducerClassesMixin, HistProducerClassMixin,
    InferenceModelMixin, HistHookMixin, MLModelsMixin, VariablesMixin,
)
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.histograms import MergeHistograms, MergeShiftedHistograms
from columnflow.histogram_io import load_histogram
from columnflow.inference import InferenceModel
from columnflow.config_util import get_datasets_from_process
from columnflow.util import dev_sandbox, DotDict, maybe_import
from columnflow.types import TYPE_CHECKING, Any

if TYPE_CHECKING:
    hist = maybe_import("hist")


class InferenceModelUser(
    InferenceModelMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # Support multiple configs.
    single_config = False

    # Upstream requirements.
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeShiftedHistograms=MergeShiftedHistograms,
    )

    _combined_config_data_attr = "_workflow_cached_combined_config_data"
    transfer_params_to_inst = {_combined_config_data_attr}

    @classmethod
    def resolve_param_values_post_init(cls, params: dict[str, Any]) -> dict[str, Any]:
        params = super().resolve_param_values_post_init(params)

        # Add combined config data to params.
        if (
            not params.get(cls._combined_config_data_attr) and
            (config_insts := params.get("config_insts")) and
            (inference_model_inst := params.get("inference_model_inst"))
        ):
            params[cls._combined_config_data_attr] = cls._combined_config_data(config_insts, inference_model_inst)

        return params

    @classmethod
    def get_mc_datasets(cls, config_inst: od.Config, proc_obj: DotDict) -> list[str]:
        """
        Helper to find MC datasets for a process object from an inference model.
        """
        if not (config_data := proc_obj.config_data.get(config_inst.name)):
            return []

        if config_data.mc_datasets:
            return [
                dataset.name
                for dataset in config_inst.datasets
                if (
                    dataset.is_mc and
                    law.util.multi_match(dataset.name, config_data.mc_datasets, mode=any)
                )
            ]

        if proc_obj.is_dynamic:
            return []

        return [
            dataset_inst.name
            for dataset_inst in get_datasets_from_process(config_inst, config_data.process)
        ]

    @classmethod
    def get_data_datasets(cls, config_inst: od.Config, cat_obj: DotDict) -> list[str]:
        """
        Helper to find data datasets for a category object from an inference model.
        """
        if not (config_data := cat_obj.config_data.get(config_inst.name)):
            return []

        if not config_data.data_datasets:
            return []

        return [
            dataset.name
            for dataset in config_inst.datasets
            if (
                dataset.is_data and
                law.util.multi_match(dataset.name, config_data.data_datasets, mode=any)
            )
        ]

    @classmethod
    def _combined_config_data(
        cls,
        config_insts: list[od.Config],
        inference_model_inst: InferenceModel,
    ) -> DotDict[od.ConfigInst, dict[str, dict | set]]:
        config_data = {
            config_inst: {
                "variables": set(),
                "categories": set(),
                "data_datasets": set(),
                "mc_datasets": {},
            }
            for config_inst in config_insts
        }

        for cat_obj in inference_model_inst.categories:
            variables = set()
            categories = set()

            # Do not narrow config_insts in place: subsequent categories may
            # apply to other configs.
            category_config_insts = [
                config_inst for config_inst in config_insts
                if config_inst.name in cat_obj.config_data
            ]

            for config_inst in category_config_insts:
                data = config_data[config_inst]

                data["variables"].add(cat_obj.config_data[config_inst.name].variable)
                data["categories"].add(cat_obj.config_data[config_inst.name].category)

                if not cat_obj.data_from_processes or any(proc_obj.is_dynamic for proc_obj in cat_obj.processes):
                    data["data_datasets"].update(cls.get_data_datasets(config_inst, cat_obj))

                for proc_obj in cat_obj.processes:
                    mc_datasets = cls.get_mc_datasets(config_inst, proc_obj)
                    for dataset_name in mc_datasets:
                        if dataset_name not in data["mc_datasets"]:
                            data["mc_datasets"][dataset_name] = {
                                "shift_sources": set(),
                                "proc_names": set(),
                            }
                        data["mc_datasets"][dataset_name]["proc_names"].add(
                            proc_obj.config_data[config_inst.name].process,
                        )

                    for param_obj in proc_obj.parameters:
                        if config_inst.name not in param_obj.config_data:
                            continue
                        if (
                            (param_obj.type.is_shape and not param_obj.transformations.any_from_rate) or
                            (param_obj.type.is_rate and param_obj.transformations.any_from_shape)
                        ):
                            shift_source = param_obj.config_data[config_inst.name].shift_source
                            for mc_dataset in mc_datasets:
                                data["mc_datasets"][mc_dataset]["shift_sources"].add(shift_source)

                variables.add(cat_obj.config_data[config_inst.name].variable)
                categories.add(cat_obj.config_data[config_inst.name].category)

            if len(variables) != 1:
                raise ValueError(
                    f"found diverging variables to be used in datacard category '{cat_obj.name}' across configs "
                    f"{', '.join(c.name for c in category_config_insts)}: {variables}",
                )
            if len(categories) != 1:
                raise ValueError(
                    f"found diverging categories to be used in datacard category '{cat_obj.name}' across configs "
                    f"{', '.join(c.name for c in category_config_insts)}: {categories}",
                )

        return DotDict.wrap(config_data)

    @law.workflow_property(cache=True)
    def combined_config_data(self) -> DotDict[od.ConfigInst, dict[str, dict | set]]:
        return self._combined_config_data(self.config_insts, self.inference_model_inst)

    def req_branch(self, branch: int, **kwargs) -> InferenceModelUser:
        kwargs.setdefault(self._combined_config_data_attr, self.combined_config_data)
        return super().req_branch(branch, **kwargs)


class _SerializeInferenceModelBase(
    CalibratorClassesMixin,
    SelectorClassMixin,
    ReducerClassMixin,
    ProducerClassesMixin,
    MLModelsMixin,
    HistProducerClassMixin,
    HistHookMixin,
    InferenceModelUser,
):
    """
    Base classes for :py:class:`SerializeInferenceModelBase`.
    """


class SerializeInferenceModelBase(_SerializeInferenceModelBase):

    # Both requirement types are necessary:
    # - data: nominal MergeHistograms, as used in PlotShiftedVariables1D;
    # - MC: MergeShiftedHistograms with configurable shift-source expansion.
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeHistograms=MergeHistograms,
        MergeShiftedHistograms=MergeShiftedHistograms,
    )

    def create_branch_map(self):
        return {0: None}

    def requires_histogram(self, **kwargs):
        """MC shifted-histogram requirement (kept for existing overrides)."""
        return self.reqs.MergeShiftedHistograms.req_different_branching(self, **kwargs)

    def get_hist_requirement_variables(self, variables: set[str]) -> set[str]:
        """Allow inference models to expand variables needed by histogram tasks."""
        expander = getattr(
            self.inference_model_inst,
            "get_hist_requirement_variables",
            None,
        )
        if callable(expander):
            return set(expander(set(variables)))
        return set(variables)

    def get_hist_requirement_producers(
        self,
        variables: set[str],
    ) -> tuple[str, ...]:
        """Allow inference models to customize producer requirements."""
        producers = tuple(self.producers)
        resolver = getattr(
            self.inference_model_inst,
            "get_hist_requirement_producers",
            None,
        )
        if callable(resolver):
            return tuple(resolver(set(variables), producers))
        return producers

    def get_hist_requirement_variables_for_dataset(
        self,
        variables: set[str],
        dataset_inst: od.Dataset,
    ) -> set[str]:
        resolver = getattr(
            self.inference_model_inst,
            "get_hist_requirement_variables_for_dataset",
            None,
        )
        if callable(resolver):
            return set(resolver(set(variables), dataset_inst))
        return self.get_hist_requirement_variables(variables)

    def get_hist_requirement_shift_sources(
        self,
        config_inst: od.Config,
        dataset_inst: od.Dataset,
        required_sources: set[str],
    ) -> tuple[str, ...]:
        """
        Allow an analysis inference model to choose a canonical MC shift-source
        superset already produced by a plotting workflow.

        An inference model without this optional method retains the original
        process-specific shift-source behavior. Additional sources in storage
        do not introduce additional nuisance parameters in the datacard.
        """
        resolver = getattr(
            self.inference_model_inst,
            "get_hist_requirement_shift_sources",
            None,
        )
        if callable(resolver):
            sources = set(resolver(config_inst, dataset_inst, set(required_sources)))
            missing_required = set(required_sources) - sources
            if missing_required:
                raise ValueError(
                    f"hist requirement shift resolver dropped required sources for '{dataset_inst.name}': "
                    f"{sorted(missing_required)}"
                )
        else:
            sources = set(required_sources)

        sources.add("nominal")
        return tuple(sorted(sources))

    def requires_histograms(self, **kwargs):
        """
        Return histogram requirements keyed by config and dataset.

        MC can reuse full-shift merged bundles when its inference model defines
        get_hist_requirement_shift_sources(). Data reuse nominal MergeHistograms.
        """
        reqs = {}
        for config_inst, data in self.combined_config_data.items():
            reqs[config_inst.name] = {}

            model_variables = set(data["variables"])
            hist_variables = self.get_hist_requirement_variables(model_variables)
            hist_producers = self.get_hist_requirement_producers(model_variables)

            if not hist_variables:
                raise ValueError(
                    f"inference model '{self.inference_model}' has no variables "
                    f"for config '{config_inst.name}'"
                )

            # Validate the variables required by the model.
            for var_name in set.union(
                *map(set, map(VariablesMixin.split_multi_variable, hist_variables))
            ):
                if not config_inst.has_variable(var_name):
                    raise ValueError(
                        f"config '{config_inst.name}' does not have variable '{var_name}' defined as required by "
                        f"inference model '{self.inference_model}'"
                    )

            # MC datasets: use an optional canonical full-source set, matching
            # the already-completed plotting MergeShiftedHistograms task identity.
            for dataset_name in sorted(data["mc_datasets"]):
                dataset_inst = config_inst.get_dataset(dataset_name)
                dataset_variables = tuple(sorted(
                    self.get_hist_requirement_variables_for_dataset(model_variables, dataset_inst)
                ))
                required_sources = set(data["mc_datasets"][dataset_name]["shift_sources"])
                shift_sources = self.get_hist_requirement_shift_sources(
                    config_inst,
                    dataset_inst,
                    required_sources,
                )
                reqs[config_inst.name][dataset_name] = self.requires_histogram(
                    config=config_inst.name,
                    dataset=dataset_name,
                    shift_sources=shift_sources,
                    variables=dataset_variables,
                    producers=hist_producers,
                    **kwargs,
                )

            # Data datasets: no shifted-merge job is needed. Plotting already
            # provides nominal histograms using MergeHistograms directly.
            for dataset_name in sorted(data["data_datasets"]):
                dataset_inst = config_inst.get_dataset(dataset_name)
                dataset_variables = tuple(sorted(
                    self.get_hist_requirement_variables_for_dataset(model_variables, dataset_inst)
                ))
                reqs[config_inst.name][dataset_name] = (
                    self.reqs.MergeHistograms.req_different_branching(
                        self,
                        config=config_inst.name,
                        dataset=dataset_name,
                        variables=dataset_variables,
                        producers=hist_producers,
                        **kwargs,
                    )
                )

        return reqs

    def workflow_requires(self):
        reqs = super().workflow_requires()
        reqs["merged_hists"] = self.requires_histograms()
        return reqs

    def requires(self):
        return self.requires_histograms(branch=-1, workflow="local")

    def record_histogram_recovery(self, action: str, **details):
        """Persist one audit record whenever an automatic recovery is applied."""
        import os
        import re
        from datetime import datetime, timezone

        log_dir = os.environ.get(
            "CF_DATACARD_RECOVERY_DIR",
            os.path.join(os.environ.get("CF_BASE", os.getcwd()), "data", "datacard_recovery_logs"),
        )
        os.makedirs(log_dir, exist_ok=True)
        task_label = re.sub(r"[^A-Za-z0-9_.-]", "_", self.task_id)[-100:]
        path = os.path.join(log_dir, f"{task_label}_pid{os.getpid()}.txt")
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "task": self.task_id,
            "inference_model": self.inference_model,
            "version": self.version,
            "status": "DATACARD_WRITTEN" if action == "DATACARD_WRITTEN" else "APPLIED_IN_MEMORY",
            **details,
        }
        # One complete line per write, with a separate file for each PID.
        with open(path, "a", encoding="utf-8") as out:
            out.write("\t".join(f"{key}={value!r}" for key, value in record.items()) + "\n")
        if action != "DATACARD_WRITTEN":
            self._histogram_recovery_applied = True
        self.logger.warning(f"[CreateDatacards] Recovery audit: {action}; file={path}")
        return path

    def load_process_hists(
        self,
        config_inst: od.Config,
        dataset_processes: dict[str, list[str]],
        variable: str,
        inputs: dict,
    ) -> dict[str, dict[od.Process, hist.Hist]]:
        import hist
        import luigi
        import numpy as np
        from columnflow.hist_util import ensure_bin_exists

        def is_zero(h):
            """Require zero sum of weights AND zero variance, including flow bins."""
            values = h.values(flow=True)
            variances = h.variances(flow=True)
            return (
                np.all(np.isfinite(values))
                and not np.any(values != 0)
                and (
                    variances is None or (
                        np.all(np.isfinite(variances)) and not np.any(variances != 0)
                    )
                )
            )

        def verify_task(task, label):
            if not task.complete():
                raise ValueError(f"[CreateDatacards] Incomplete {label}: {task.task_id}")

        hists: dict[od.Process, hist.Hist] = {}
        nominal_shift = config_inst.get_shift("nominal").name

        with self.publish_step(f"extracting '{variable}' for config {config_inst.name} ..."):
            for dataset_name, process_names in dataset_processes.items():
                collection = inputs[dataset_name]["collection"].targets
                if not collection:
                    raise ValueError(
                        f"[CreateDatacards] No histogram input branches: "
                        f"{config_inst.name}, {dataset_name}, {variable}"
                    )

                for branch, inp in collection.items():
                    try:
                        h = load_histogram(inp["hists"], variable, config_inst, dataset=dataset_name)
                    except Exception as e:
                        raise ValueError(
                            f"[CreateDatacards] Cannot load {config_inst.name}, "
                            f"{dataset_name}, {variable}, branch={branch}: {e}"
                        ) from e

                    # Account for every branch. Only a truly empty branch is skippable.
                    statuses = getattr(self, "_dataset_input_status", {})
                    input_key = (config_inst.name, variable, dataset_name)
                    status = statuses.setdefault(input_key, {"checked": 0, "empty": 0})
                    status["checked"] += 1
                    self._dataset_input_status = statuses

                    if "shift" not in h.axes.name:
                        raise ValueError(
                            f"[CreateDatacards] Missing shift axis: "
                            f"{config_inst.name}, {dataset_name}, {variable}, "
                            f"axes={list(h.axes.name)}"
                        )
                    available_shifts = list(h.axes["shift"])
                    if nominal_shift not in available_shifts:
                        if not available_shifts and is_zero(h):
                            status["empty"] += 1
                            self.logger.warning(
                                f"[CreateDatacards] Skipping verified empty input: "
                                f"config={config_inst.name}, dataset={dataset_name}, "
                                f"variable={variable}, branch={branch}"
                            )
                            continue

                        # Check the exact nominal dependency, not an unrelated file.
                        shifted_workflow = self.requires()[config_inst.name][dataset_name]
                        shifted_branch = shifted_workflow.req_branch(branch)
                        upstream_reqs = shifted_branch.requires()
                        nominal_task = upstream_reqs.get(nominal_shift)
                        if nominal_task is None:
                            raise ValueError(
                                f"[CreateDatacards] Missing nominal task dependency for "
                                f"{config_inst.name}, {dataset_name}, {variable}, branch={branch}"
                            )
                        verify_task(nominal_task, "nominal MergeHistograms")
                        upstream_deps = list(luigi.task.flatten(nominal_task.requires()))
                        if not upstream_deps:
                            raise ValueError(
                                f"[CreateDatacards] No nominal upstream dependencies for "
                                f"{config_inst.name}, {dataset_name}, {variable}"
                            )
                        for dep in upstream_deps:
                            verify_task(dep, "nominal histogram dependency")

                        try:
                            nominal_input = load_histogram(
                                nominal_task.output()["hists"], variable,
                                config_inst, dataset=dataset_name,
                            )
                        except Exception as e:
                            raise ValueError(
                                f"[CreateDatacards] Cannot load upstream nominal "
                                f"MergeHistograms for {dataset_name}, {variable}: {e}"
                            ) from e
                        if "shift" not in nominal_input.axes.name:
                            raise ValueError(
                                f"[CreateDatacards] Upstream nominal histogram has no shift axis: "
                                f"{dataset_name}, {variable}"
                            )
                        nominal_upstream_shifts = list(nominal_input.axes["shift"])
                        nominal_source = (
                            nominal_input[{"shift": hist.loc(nominal_shift)}]
                            if nominal_shift in nominal_upstream_shifts
                            else nominal_input
                        )
                        if not is_zero(nominal_source):
                            raise ValueError(
                                f"[CreateDatacards] Refusing zero recovery: upstream nominal "
                                f"has entries or nonzero variances; config={config_inst.name}, "
                                f"dataset={dataset_name}, variable={variable}, "
                                f"branch={branch}, shifts={nominal_upstream_shifts}"
                            )

                        # Save the contents and variances of every existing shift.
                        original = {}
                        for shift in available_shifts:
                            part = h[{"shift": hist.loc(shift)}]
                            variance = part.variances(flow=True)
                            original[shift] = (
                                np.array(part.values(flow=True), copy=True),
                                None if variance is None else np.array(variance, copy=True),
                            )

                        h = ensure_bin_exists(h, "shift", nominal_shift)
                        added_weight_shifts = []
                        added_external_shifts = []

                        # Embedded weight variations derive from nominal-selected events.
                        # If the nominal selection has zero entries, these are zero too.
                        for source in getattr(config_inst.x, "histogram_weight_shift_sources", ()):
                            for direction in ("up", "down"):
                                shift = f"{source}_{direction}"
                                if not config_inst.has_shift(shift) or shift in h.axes["shift"]:
                                    continue
                                h = ensure_bin_exists(h, "shift", shift)
                                added_weight_shifts.append(shift)

                        # External shifts are independently verified against their own
                        # complete upstream MergeHistograms output.
                        for shift, task in upstream_reqs.items():
                            if shift == nominal_shift or shift in h.axes["shift"]:
                                continue
                            verify_task(task, f"{shift} MergeHistograms")
                            try:
                                upstream_hist = load_histogram(
                                    task.output()["hists"], variable,
                                    config_inst, dataset=dataset_name,
                                )
                            except Exception as e:
                                raise ValueError(
                                    f"[CreateDatacards] Cannot load upstream shift {shift} "
                                    f"for {dataset_name}, {variable}: {e}"
                                ) from e
                            if "shift" not in upstream_hist.axes.name:
                                raise ValueError(
                                    f"[CreateDatacards] Upstream shift {shift} "
                                    f"has no shift axis for {dataset_name}"
                                )
                            upstream_names = list(upstream_hist.axes["shift"])
                            upstream_slice = (
                                upstream_hist[{"shift": hist.loc(shift)}]
                                if shift in upstream_names else upstream_hist
                            )
                            if not is_zero(upstream_slice):
                                raise ValueError(
                                    f"[CreateDatacards] Shift {shift} missing in shifted merge, "
                                    f"but its upstream input is nonempty: "
                                    f"{config_inst.name}, {dataset_name}, {variable}"
                                )
                            h = ensure_bin_exists(h, "shift", shift)
                            added_external_shifts.append(shift)

                        reconstructed = [nominal_shift] + added_weight_shifts + added_external_shifts
                        for shift in reconstructed:
                            if not is_zero(h[{"shift": hist.loc(shift)}]):
                                raise ValueError(
                                    f"[CreateDatacards] Reconstructed shift is nonzero: "
                                    f"{shift}, {config_inst.name}, {dataset_name}, {variable}"
                                )

                        for shift, (old_values, old_variances) in original.items():
                            part = h[{"shift": hist.loc(shift)}]
                            new_var = part.variances(flow=True)
                            if (
                                not np.array_equal(old_values, part.values(flow=True), equal_nan=True)
                                or (old_variances is None) != (new_var is None)
                                or (
                                    old_variances is not None and
                                    not np.array_equal(old_variances, new_var, equal_nan=True)
                                )
                            ):
                                raise ValueError(
                                    f"[CreateDatacards] Recovery modified original shift {shift}: "
                                    f"{config_inst.name}, {dataset_name}, {variable}"
                                )

                        self.record_histogram_recovery(
                            "ZERO_NOMINAL_RECOVERED",
                            config=config_inst.name,
                            dataset=dataset_name,
                            variable=variable,
                            branch=branch,
                            upstream_nominal_task=nominal_task.task_id,
                            upstream_nominal_verified_empty=True,
                            nominal_dependencies_verified_complete=True,
                            original_shifts=available_shifts,
                            added_nominal=nominal_shift,
                            added_weight_shifts=added_weight_shifts,
                            added_external_shifts=added_external_shifts,
                            original_shift_values_and_variances_preserved=True,
                            statistical_review_required=True,
                        )

                    process_insts = [config_inst.get_process(name) for name in process_names]
                    for process_inst in process_insts:
                        descendants = [
                            sub for sub, _, _ in process_inst.walk_processes(include_self=True)
                        ]
                        if len(h.axes["process"]):
                            selected = [sub.name for sub in descendants if sub.name in h.axes["process"]]
                            if not selected:
                                raise ValueError(
                                    f"[CreateDatacards] Process {process_inst.name} not present "
                                    f"in nonempty dataset input {dataset_name}, variable {variable}"
                                )
                            h_proc = h[{"process": [hist.loc(name) for name in selected]}]
                            h_proc = h_proc[{"process": sum}]
                        else:
                            h_proc = h[{"process": sum}]
                        h_proc = self.modify_process_hist(
                            config_inst=config_inst,
                            process_inst=process_inst,
                            variable=variable,
                            h=h_proc,
                        )
                        if process_inst in hists:
                            hists[process_inst] += h_proc
                        else:
                            hists[process_inst] = h_proc

        return hists

    def modify_process_hist(
        self,
        config_inst: od.Config,
        process_inst: od.Process,
        variable: str,
        h: hist.Hist,
    ) -> hist.Hist:
        """Hook to modify a process histogram after loading."""
        return h