# coding: utf-8

"""
Tasks related to the creation of datacards for inference purposes.
"""

from __future__ import annotations

import collections

import law
import order as od

from columnflow.tasks.framework.base import AnalysisTask, wrapper_factory
from columnflow.tasks.framework.inference import SerializeInferenceModelBase
from columnflow.tasks.histograms import MergeHistograms
from columnflow.inference.cms.datacard import DatacardWriter
from columnflow.hist_util import select_category_bins
from columnflow.types import TYPE_CHECKING

if TYPE_CHECKING:
    from columnflow.inference.cms.datacard import DatacardHists, ShiftHists


class CreateDatacards(SerializeInferenceModelBase):

    resolution_task_cls = MergeHistograms
    datacard_writer_cls = DatacardWriter

    def verify_missing_systematic_is_zero(
        self,
        config_inst,
        process_inst,
        expected_datasets,
        variable,
        category_inst,
        shift_name,
        inputs,
    ):
        """
        Check the exact upstream MergeHistograms outputs for a missing external
        systematic, restricted to the same process and datacard category.

        Return the checked upstream task IDs only when EVERY expected dataset
        has the required shift dependency and that process/category has
        exactly zero weight and zero statistical variance in all its branches.
        Never use the nominal histogram as evidence for an external shift.
        """
        import hist
        import numpy as np

        from columnflow.histogram_io import load_histogram

        if not expected_datasets:
            raise ValueError(
                f"[CreateDatacards] Cannot verify missing shift '{shift_name}' "
                f"for process '{process_inst.name}': no source MC datasets found"
            )

        task_ids = []
        process_names = {
            sub.name
            for sub, _, _ in process_inst.walk_processes(include_self=True)
        }

        for dataset_name in sorted(expected_datasets):
            shifted_workflow = self.requires()[config_inst.name][dataset_name]
            branches = inputs[config_inst.name][dataset_name]["collection"].targets

            if not branches:
                raise ValueError(
                    f"[CreateDatacards] No branches to verify '{shift_name}' "
                    f"for {config_inst.name}, {dataset_name}, {variable}"
                )

            checked_this_dataset = 0

            for branch in branches:
                shifted_branch = shifted_workflow.req_branch(branch)
                upstream_reqs = shifted_branch.requires()
                upstream_task = upstream_reqs.get(shift_name)

                # With shift-source splitting, this branch might cover
                # another set of systematic shifts.
                if upstream_task is None:
                    continue

                checked_this_dataset += 1
                if not upstream_task.complete():
                    raise ValueError(
                        f"[CreateDatacards] Upstream task for '{shift_name}' "
                        f"is incomplete: {upstream_task.task_id}"
                    )

                try:
                    upstream_hist = load_histogram(
                        upstream_task.output()["hists"],
                        variable,
                        config_inst,
                        dataset=dataset_name,
                    )
                except Exception as exc:
                    raise ValueError(
                        f"[CreateDatacards] Failed to inspect upstream "
                        f"'{shift_name}' for {dataset_name}, {variable}: {exc}"
                    ) from exc

                axis_names = upstream_hist.axes.name
                if not all(axis in axis_names for axis in ("process", "category", "shift")):
                    raise ValueError(
                        f"[CreateDatacards] Cannot verify '{shift_name}': "
                        f"upstream histogram has axes {list(axis_names)}"
                    )

                # A process absent from the categorical process axis has no
                # contribution in this particular upstream histogram.
                present_processes = [
                    name for name in upstream_hist.axes["process"]
                    if name in process_names
                ]
                if not present_processes:
                    task_ids.append(upstream_task.task_id)
                    continue

                subhist = upstream_hist[{
                    "process": [hist.loc(name) for name in present_processes],
                }]
                subhist = subhist[{"process": sum}]
                subhist = select_category_bins(
                    subhist, category_inst,
                    use_leaves=True, prefer_parents=True, reduce=True,
                )

                if shift_name in subhist.axes["shift"]:
                    subhist = subhist[{"shift": hist.loc(shift_name)}]
                # If this shift has no categorical bin, only a completely
                # zero process/category slice certifies absence of content.

                values = np.asarray(subhist.values(flow=True))
                variances = subhist.variances(flow=True)
                if (
                    variances is None
                    or not np.all(np.isfinite(values))
                    or not np.all(np.isfinite(variances))
                    or np.any(values != 0)
                    or np.any(variances != 0)
                ):
                    raise ValueError(
                        f"[CreateDatacards] Missing '{shift_name}' for "
                        f"{config_inst.name}, process={process_inst.name}, "
                        f"dataset={dataset_name}, category={category_inst.name}, "
                        f"variable={variable}; its upstream process/category "
                        f"is nonzero, nonfinite, or lacks sumw2. "
                        f"Refusing to invent the systematic template."
                    )

                task_ids.append(upstream_task.task_id)

            if checked_this_dataset == 0:
                raise ValueError(
                    f"[CreateDatacards] No upstream MergeHistograms requirement "
                    f"for '{shift_name}' in {config_inst.name}, "
                    f"dataset={dataset_name}, variable={variable}. "
                    f"Cannot establish whether it is zero."
                )

        return task_ids

    def output(self):
        def basename(cat_obj, name, ext):
            parts = [name, cat_obj.name]
            if (hooks_repr := self.hist_hooks_repr):
                parts.append(f"hooks_{hooks_repr}")
            if cat_obj.postfix is not None:
                parts.append(cat_obj.postfix)
            return f"{'__'.join(map(str, parts))}.{ext}"

        return law.SiblingFileCollection({
            cat_obj.name: {
                "card": self.target(basename(cat_obj, "datacard", "txt")),
                "shapes": self.target(basename(cat_obj, "shapes", "root")),
            }
            for cat_obj in self.inference_model_inst.categories
        })

    @law.decorator.log
    @law.decorator.safe_output
    def run(self):
        import hist
        import numpy as np

        inputs = self.input()
        outputs = self.output()
        self._histogram_recovery_applied = False

        # Step 1: group the requested variables and configs.
        variable_data = collections.defaultdict(set)
        for config_inst, data in self.combined_config_data.items():
            for variable in data["variables"]:
                variable_data[variable].add(config_inst)

        for variable, variable_config_insts in variable_data.items():

            # Keep input status for every expected dataset and input branch.
            self._dataset_input_status = {}

            # Step 2: load histograms for each relevant config.
            input_hists: dict[od.Config, dict[od.Process, hist.Hist]] = {}
            for config_inst in variable_config_insts:
                data = self.combined_config_data[config_inst]
                input_hists[config_inst] = self.load_process_hists(
                    config_inst,
                    {
                        dataset_name: list(data["mc_datasets"][dataset_name]["proc_names"])
                        for dataset_name in data["mc_datasets"]
                    } | {
                        dataset_name: ["data"]
                        for dataset_name in data["data_datasets"]
                    },
                    variable,
                    inputs[config_inst.name],
                )

            # Step 3: build each datacard category.
            for cat_obj in self.inference_model_inst.categories:
                if not any(d.variable == variable for d in cat_obj.config_data.values()):
                    continue
                if not all(d.variable == variable for d in cat_obj.config_data.values()):
                    raise ValueError(
                        f"datacard category '{cat_obj.name}' uses different variables for its configs: "
                        f"{[d.variable for d in cat_obj.config_data.values()]}"
                    )

                config_insts = [
                    cfg for cfg in self.config_insts if cfg.name in cat_obj.config_data
                ]
                if not config_insts:
                    continue
                self.publish_message(
                    f"processing inputs for category '{cat_obj.name}' with variable '{variable}'"
                )
                category = cat_obj.config_data[config_insts[0].name].category

                # Step 4: apply histogram hooks, including the QCD estimate.
                _input_hists = self.invoke_hist_hooks(
                    {cfg: input_hists[cfg].copy() for cfg in config_insts},
                    hook_kwargs={"variable_name": variable, "category_name": category},
                )

                # Step 5: make the structures expected by DatacardWriter.
                datacard_hists: DatacardHists = {cat_obj.name: {}}
                zero_process_specs = []

                for config_inst in _input_hists.keys():
                    config_data = cat_obj.config_data.get(config_inst.name)
                    category_inst = config_inst.get_category(category)

                    if cat_obj.data_from_processes:
                        for process_inst in list(_input_hists[config_inst]):
                            if process_inst.is_data:
                                del _input_hists[config_inst][process_inst]

                    proc_objs = list(cat_obj.processes)
                    if config_data.data_datasets and not cat_obj.data_from_processes:
                        proc_objs.append(self.inference_model_inst.process_spec(name="data"))

                    for proc_obj in proc_objs:
                        if config_inst.name not in proc_obj.config_data and proc_obj.name != "data":
                            continue

                        if proc_obj.is_dynamic:
                            if not (process_name := proc_obj.config_data[config_inst.name].get("process", None)):
                                raise ValueError(
                                    f"dynamic datacard process object misses 'process' entry in config data for "
                                    f"'{config_inst.name}': {proc_obj}"
                                )
                            process_inst = config_inst.get_process(process_name)
                        else:
                            process_inst = config_inst.get_process(
                                proc_obj.name if proc_obj.name == "data"
                                else proc_obj.config_data[config_inst.name].process,
                            )

                        h_proc = _input_hists[config_inst].get(process_inst, None)
                        zero_template = False
                        if h_proc is None:
                            expected_datasets = (
                                set(self.get_mc_datasets(config_inst, proc_obj))
                                if proc_obj.name != "data" and not proc_obj.is_dynamic
                                else set()
                            )
                            statuses = getattr(self, "_dataset_input_status", {})
                            input_checks = {
                                dataset: statuses.get((config_inst.name, variable, dataset), {})
                                for dataset in expected_datasets
                            }
                            all_inputs_empty = bool(expected_datasets) and all(
                                status.get("checked", 0) > 0
                                and status["checked"] == status.get("empty", -1)
                                for status in input_checks.values()
                            )
                            if not all_inputs_empty:
                                raise ValueError(
                                    f"\n[CreateDatacards] Missing required process histogram\n"
                                    f"  Config:            {config_inst.name}\n"
                                    f"  Category:          {cat_obj.name}\n"
                                    f"  Process:           {proc_obj.name}\n"
                                    f"  Variable:          {variable}\n"
                                    f"  Expected datasets: {sorted(expected_datasets)}\n"
                                    f"  Input checks:      {input_checks}\n"
                                    f"  All inputs empty:  {all_inputs_empty}\n"
                                    f"  Inference model:   {self.inference_model}\n"
                                    f"Unexpected or unchecked input: refusing zero-template recovery."
                                )

                            # The dataset outputs were loaded successfully and
                            # every expected branch is confirmed completely empty.
                            donor = next(
                                (
                                    h for h in _input_hists[config_inst].values()
                                    if "shift" in h.axes.name and "nominal" in h.axes["shift"]
                                ),
                                None,
                            )
                            if donor is None:
                                raise ValueError(
                                    f"[CreateDatacards] No compatible donor for zero template: "
                                    f"{config_inst.name}, {proc_obj.name}, {variable}"
                                )
                            h_proc = donor.copy().reset()
                            zero_template = True

                            # DatacardWriter normally drops empty nominal shapes.
                            # The zero process is retained explicitly for review.
                            if getattr(proc_obj, "skip_if_empty", False):
                                zero_process_specs.append((proc_obj, proc_obj.skip_if_empty))
                                proc_obj.skip_if_empty = False

                            self.record_histogram_recovery(
                                "ZERO_PROCESS_RECOVERED",
                                config=config_inst.name,
                                category=cat_obj.name,
                                process=proc_obj.name,
                                variable=variable,
                                expected_datasets=sorted(expected_datasets),
                                input_checks=input_checks,
                                all_inputs_verified_empty=True,
                                statistical_review_required=True,
                            )

                        h_proc = select_category_bins(
                            h_proc, category_inst, use_leaves=True,
                            prefer_parents=True, reduce=True,
                        )

                        datacard_hists[cat_obj.name].setdefault(proc_obj.name, {}).setdefault(
                            config_inst.name, {}
                        )
                        shift_hists: ShiftHists = datacard_hists[cat_obj.name][proc_obj.name][config_inst.name]
                        nominal_shift = config_inst.get_shift("nominal").name
                        available_shifts = (
                            list(h_proc.axes["shift"]) if "shift" in h_proc.axes.name else []
                        )
                        if nominal_shift not in available_shifts:
                            raise ValueError(
                                f"\n[CreateDatacards] Missing nominal histogram\n"
                                f"  Config:   {config_inst.name}\n"
                                f"  Category: {cat_obj.name}\n"
                                f"  Process:  {proc_obj.name}\n"
                                f"  Variable: {variable}\n"
                                f"  Available shifts: {available_shifts}\n"
                            )

                        nominal_hist = h_proc[{"shift": hist.loc(nominal_shift)}]
                        shift_hists["nominal"] = nominal_hist

                        # Flag nominal-zero/systematic-nonzero combinations.
                        if not zero_template and proc_obj.name != "data":
                            nominal_var = nominal_hist.variances(flow=True)
                            nominal_is_zero = (
                                not np.any(nominal_hist.values(flow=True) != 0)
                                and (nominal_var is None or not np.any(nominal_var != 0))
                            )
                            if nominal_is_zero:
                                nonzero_variations = [
                                    shift for shift in available_shifts
                                    if shift != nominal_shift
                                    and np.any(
                                        h_proc[{"shift": hist.loc(shift)}].values(flow=True) != 0
                                    )
                                ]
                                if nonzero_variations:
                                    self.record_histogram_recovery(
                                        "ZERO_NOMINAL_NONZERO_VARIATIONS_REVIEW",
                                        config=config_inst.name,
                                        category=cat_obj.name,
                                        process=proc_obj.name,
                                        variable=variable,
                                        shifts=nonzero_variations,
                                        process_skip_if_empty=proc_obj.skip_if_empty,
                                        statistical_review_required=True,
                                    )

                        if zero_template:
                            for param_obj in proc_obj.parameters:
                                need_shapes = (
                                    (param_obj.type.is_shape and not param_obj.transformations.any_from_rate) or
                                    (param_obj.type.is_rate and param_obj.transformations.any_from_shape)
                                )
                                if not need_shapes:
                                    continue
                                for direction in ("up", "down"):
                                    shift_hists[(param_obj.name, direction)] = nominal_hist.copy()
                            continue

                        if proc_obj.name == "data":
                            continue

                        for param_obj in proc_obj.parameters:
                            need_shapes = (
                                (param_obj.type.is_shape and not param_obj.transformations.any_from_rate) or
                                (param_obj.type.is_rate and param_obj.transformations.any_from_shape)
                            )
                            if not need_shapes:
                                continue
                            shift_source = (
                                param_obj.config_data[config_inst.name].shift_source
                                if config_inst.name in param_obj.config_data else None
                            )
                            for direction in ("up", "down"):
                                shift = f"{shift_source}_{direction}" if shift_source else nominal_shift
                                if shift not in h_proc.axes["shift"]:
                                    if proc_obj.is_dynamic:
                                        raise ValueError(
                                            f"[CreateDatacards] Cannot recover '{shift}' "
                                            f"for dynamic process '{proc_obj.name}'"
                                        )

                                    weight_sources = set(
                                        config_inst.x("histogram_weight_shift_sources", ())
                                    )

                                    if shift_source in weight_sources:
                                        # Embedded weight variations are filled
                                        # from the nominal event selection. Only
                                        # a zero nominal WITH zero sumw2 is safe.
                                        values = np.asarray(nominal_hist.values(flow=True))
                                        variances = nominal_hist.variances(flow=True)
                                        if (
                                            variances is None
                                            or not np.all(np.isfinite(values))
                                            or not np.all(np.isfinite(variances))
                                            or np.any(values != 0)
                                            or np.any(variances != 0)
                                        ):
                                            raise ValueError(
                                                f"[CreateDatacards] Missing weight shift "
                                                f"'{shift}' for nonempty or unverified "
                                                f"nominal process '{proc_obj.name}' "
                                                f"in {config_inst.name}, {variable}"
                                            )
                                        evidence = "zero_nominal_contents_and_sumw2"
                                        checked_tasks = []
                                    else:
                                        expected_datasets = self.get_mc_datasets(
                                            config_inst, proc_obj
                                        )
                                        checked_tasks = self.verify_missing_systematic_is_zero(
                                            config_inst=config_inst,
                                            process_inst=process_inst,
                                            expected_datasets=expected_datasets,
                                            variable=variable,
                                            category_inst=category_inst,
                                            shift_name=shift,
                                            inputs=inputs,
                                        )
                                        evidence = "all_upstream_process_category_shapes_zero"

                                    # Never substitute nominal contents for a
                                    # missing variation. Use an explicit zero
                                    # histogram only after verification.
                                    shift_hists[(param_obj.name, direction)] = (
                                        nominal_hist.copy().reset()
                                    )
                                    self.record_histogram_recovery(
                                        "ZERO_SYSTEMATIC_RECOVERED",
                                        config=config_inst.name,
                                        category=cat_obj.name,
                                        process=proc_obj.name,
                                        variable=variable,
                                        systematic=shift,
                                        nuisance=param_obj.name,
                                        direction=direction,
                                        evidence=evidence,
                                        checked_upstream_tasks=checked_tasks,
                                        statistical_review_required=True,
                                    )
                                    continue

                                shift_hists[(param_obj.name, direction)] = h_proc[{
                                    "shift": hist.loc(shift),
                                }]

                # Step 6: write the datacard and the ROOT shapes.
                outp = outputs[cat_obj.name]
                writer = self.datacard_writer_cls(self.inference_model_inst, datacard_hists)
                try:
                    with outp["card"].localize("w") as tmp_card, outp["shapes"].localize("w") as tmp_shapes:
                        writer.write(
                            tmp_card.abspath, tmp_shapes.abspath,
                            shapes_path_ref=outp["shapes"].basename,
                        )
                finally:
                    for spec, original_skip_setting in zero_process_specs:
                        spec.skip_if_empty = original_skip_setting
                self.publish_message(f"datacard written to {outp['card'].abspath}")
                if getattr(self, "_histogram_recovery_applied", False):
                    self.record_histogram_recovery(
                        "DATACARD_WRITTEN",
                        category=cat_obj.name,
                        variable=variable,
                        datacard=outp["card"].abspath,
                        shapes=outp["shapes"].abspath,
                    )
                del _input_hists
            del input_hists


CreateDatacardsWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=CreateDatacards,
    enable=["configs", "skip_configs"],
)