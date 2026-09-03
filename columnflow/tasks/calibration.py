# coding: utf-8

"""
Tasks related to calibrating events.
"""

import luigi
import law

from columnflow.tasks.framework.base import Requirements, AnalysisTask, wrapper_factory
from columnflow.tasks.framework.mixins import CalibratorMixin, ChunkedIOMixin
from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.framework.decorators import on_failure
from columnflow.tasks.external import GetDatasetLFNs
from columnflow.util import maybe_import, ensure_proxy, dev_sandbox

ak = maybe_import("awkward")


class _CalibrateEvents(
    CalibratorMixin,
    ChunkedIOMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    """
    Base classes for :py:class:`CalibrateEvents`.
    """


class CalibrateEvents(_CalibrateEvents):
    """
    Task to apply calibrations to objects, e.g. leptons and jets.

    The calibrations that are to be applied can be specified on the command line, and are
    implemented as instances of the :py:class:`~columnflow.calibration.Calibrator` class. For
    further information, please consider the documentation there.
    """

    # default sandbox, might be overwritten by calibrator function
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        GetDatasetLFNs=GetDatasetLFNs,
    )

    invokes_calibrator = True

    def workflow_requires(self) -> dict:
        """
        Configure the requirements for the workflow in general. For more general informations, see
        :external+law:py:meth:`BaseWorkflow.workflow_requires() <law.workflow.base.BaseWorkflow.workflow_requires>`.

        :return: Dictionary containing the requirements for this task.
        """
        reqs = super().workflow_requires()

        reqs["lfns"] = self.reqs.GetDatasetLFNs.req(self)

        # add calibrator dependent requirements
        reqs["calibrator"] = law.util.make_unique(law.util.flatten(self.calibrator_inst.run_requires(task=self)))

        return reqs

    def requires(self) -> dict:
        """
        Configure the requirements for the individual branches of the workflow.
        """
        reqs = {"lfns": self.reqs.GetDatasetLFNs.req(self)}

        # add calibrator dependent requirements
        reqs["calibrator"] = law.util.make_unique(law.util.flatten(self.calibrator_inst.run_requires(task=self)))

        return reqs

    def output(self):
        """
        Defines the outputs of the current branch within the workflow.
        """
        outputs = {}

        # only declare the output in case the calibrator actually creates columns
        if self.calibrator_inst.produced_columns:
            outputs["columns"] = self.target(f"calib_{self.branch}.parquet")

        return outputs

    @law.decorator.notify
    @law.decorator.log
    @ensure_proxy
    @law.decorator.localize(input=False)
    @law.decorator.safe_output
    @on_failure(callback=lambda task: task.teardown_calibrator_inst())
    def run(self):
        """
        Run method of this task.
        """
        from columnflow.columnar_util import (
            Route,
            RouteFilter,
            mandatory_coffea_columns,
            sorted_ak_to_parquet,
            update_ak_array,
            add_ak_aliases,
        )

        # prepare inputs and outputs
        lfn_task = self.requires()["lfns"]
        output = self.output()
        output_chunks = {}

        # run the calibrator setup
        self._array_function_post_init()
        calibrator_reqs = self.calibrator_inst.run_requires(task=self)
        reader_targets = self.calibrator_inst.run_setup(
            task=self,
            reqs=calibrator_reqs,
            inputs=luigi.task.getpaths(calibrator_reqs),
        )

        # create a temp dir for saving intermediate files
        tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
        tmp_dir.touch()

        # get aliases for the shift implemented by this CalibrateEvents task
        aliases = self.local_shift_inst.x("column_aliases", {})

        # define columns that need to be read
        read_columns = set(map(Route, mandatory_coffea_columns))
        read_columns |= self.calibrator_inst.used_columns

        # -------------------------------------------------------------------------
        # Define columns that will be written.
        #
        # The calibrator can internally produce all systematic variations, e.g.
        #
        #   Jet.pt_jec_Regrouped_Absolute_up
        #   Jet.pt_jec_Regrouped_Absolute_down
        #   Jet.pt_jer_up
        #   ...
        #
        # However, CalibrateEvents now owns these shifts.  Therefore, each shifted
        # task writes only the values belonging to its own shift under the nominal
        # column name.
        #
        # No JEC/JER suffixed columns are persisted.
        # -------------------------------------------------------------------------

        write_columns = set(self.calibrator_inst.produced_columns)

        # Names of all shifts implemented by this calibrator.
        #
        # A produced column ending in one of these shift names is an internal
        # systematic helper and must not be written directly.
        local_shift_suffixes = tuple(
            f"_{shift_name}"
            for shift_name in sorted(self.calibrator_inst.all_shifts)
            if shift_name != "nominal"
        )

        if local_shift_suffixes:
            write_columns = {
                route
                for route in write_columns
                if not Route(route).column.endswith(local_shift_suffixes)
            }

        # The destinations of the aliases (Jet.pt, Jet.mass, PuppiMET.pt, ...)
        # must be present in the output.
        write_columns |= {
            Route(dst)
            for dst in aliases
        }

        route_filter = RouteFilter(keep=write_columns)

        # let the lfn_task locate and prepare the nano file(s)
        nano_input = [nano_target for _, nano_target in lfn_task.iter_nano_files(self)]
        if len(nano_input) == 1:
            nano_input = nano_input[0]

        # prepare inputs for localization
        with law.localize_file_targets(
            [nano_input, *reader_targets.values()],
            mode="r",
        ) as inps:
            # iterate over chunks
            for (events, *cols), pos in self.iter_chunked_io(
                law.util.map_struct(law.target.file.get_path, inps),
                source_type=["coffea_root"] + (len(inps) - 1) * [None],
                open_options=self.get_open_options(inps, first_is_nano=True),
                read_options=self.get_read_options(inps, first_is_nano=True),
                read_columns=len(inps) * [read_columns],
                filter_config=self.get_filter_configs(inps, first_is_nano=True),
                chunk_size=self.calibrator_inst.get_min_chunk_size(),
            ):
                # adjust if necessary
                if callable(self.adjust_chunks):
                    events, *cols = self.adjust_chunks([events, *cols])

                # optional check for overlapping inputs
                if self.check_overlapping_inputs:
                    self.raise_if_overlapping([events, *cols])

                # insert additional columns
                events = update_ak_array(events, *cols)

                # invoke the calibration function
                events = self.calibrator_inst(events, task=self)

                # -------------------------------------------------------------------------
                # For a shifted calibration task, replace the nominal columns with the
                # corresponding shifted columns.
                #
                # Example:
                #
                #   local shift:
                #       jec_Regrouped_Absolute_up
                #
                #   before:
                #       Jet.pt
                #       Jet.pt_jec_Regrouped_Absolute_up
                #
                #   after:
                #       Jet.pt = old Jet.pt_jec_Regrouped_Absolute_up
                #
                # The source column is removed.
                # -------------------------------------------------------------------------

                if aliases:
                    events = add_ak_aliases(
                        events,
                        aliases,
                        remove_src=True,
                        missing_strategy="raise",
                    )

                # keep only the columns belonging to this calibration output
                events = route_filter(events)

                # optional check for finite values
                if self.check_finite_output:
                    self.raise_if_not_finite(events)

                # save as parquet via a thread in the same pool
                chunk = tmp_dir.child(f"file_{pos.index}.parquet", type="f")
                output_chunks[pos.index] = chunk
                self.chunked_io.queue(sorted_ak_to_parquet, (events, chunk.abspath))

        # teardown the calibrator
        self.teardown_calibrator_inst()

        # merge output files
        sorted_chunks = [output_chunks[key] for key in sorted(output_chunks)]
        law.pyarrow.merge_parquet_task(
            task=self,
            inputs=sorted_chunks,
            output=output["columns"],
            local=True,
            writer_opts=self.get_parquet_writer_opts(),
            target_row_group_size=self.merging_row_group_size,
        )


# overwrite class defaults
check_finite_tasks = law.config.get_expanded("analysis", "check_finite_output", [], split_csv=True)
CalibrateEvents.check_finite_output = ChunkedIOMixin.check_finite_output.copy(
    default=CalibrateEvents.task_family in check_finite_tasks,
    add_default_to_description=True,
)

check_overlap_tasks = law.config.get_expanded("analysis", "check_overlapping_inputs", [], split_csv=True)
CalibrateEvents.check_overlapping_inputs = ChunkedIOMixin.check_overlapping_inputs.copy(
    default=CalibrateEvents.task_family in check_overlap_tasks,
    add_default_to_description=True,
)


CalibrateEventsWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=CalibrateEvents,
    enable=["configs", "skip_configs", "datasets", "skip_datasets", "shifts", "skip_shifts"],
    docs="""
Wrapper task to calibrate events for multiple datasets.

:enables: ["configs", "skip_configs", "datasets", "skip_datasets", "shifts", "skip_shifts"]
""",
)
