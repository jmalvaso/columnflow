
"""
Task to produce and merge histograms.
"""

from __future__ import annotations

import luigi
import law

from columnflow.tasks.framework.base import Requirements, AnalysisTask, DatasetTask, wrapper_factory
from columnflow.tasks.framework.mixins import (
    CalibratorsMixin, SelectorStepsMixin, ProducersMixin, MLModelsMixin, VariablesMixin,
    ShiftSourcesMixin, WeightProducerMixin, ChunkedIOMixin, DatasetsProcessesMixin, CategoriesMixin
)
from columnflow.tasks.framework.plotting import ProcessPlotSettingMixin

from columnflow.tasks.framework.remote import RemoteWorkflow
from columnflow.tasks.framework.parameters import last_edge_inclusive_inst
from columnflow.tasks.reduction import ReducedEventsUser
from columnflow.tasks.production import ProduceColumns
from columnflow.tasks.ml import MLEvaluation
from columnflow.util import dev_sandbox, DotDict


class PrepareFakeFactorHistograms(
    CategoriesMixin,
    WeightProducerMixin,
    MLModelsMixin,
    ProducersMixin,
    ReducedEventsUser,
    ChunkedIOMixin,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    last_edge_inclusive = last_edge_inclusive_inst

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # upstream requirements
    reqs = Requirements(
        ReducedEventsUser.reqs,
        RemoteWorkflow.reqs,
        ProduceColumns=ProduceColumns,
    )

    # strategy for handling missing source columns when adding aliases on event chunks
    missing_column_alias_strategy = "original"

    # names of columns that contain category ids
    # (might become a parameter at some point)
    category_id_columns = {"category_ids"}

    # register sandbox and shifts found in the chosen weight producer to this task
    register_weight_producer_sandbox = True
    register_weight_producer_shifts = True

    @law.util.classproperty
    def mandatory_columns(cls) -> set[str]:
        return set(cls.category_id_columns) | {"process_id"}

    # def create_branch_map(self):
    #     # create a dummy branch map so that this task could be submitted as a job
    #     return {0: None}
    
    def workflow_requires(self):
        reqs = super().workflow_requires()

        # require the full merge forest
        reqs["events"] = self.reqs.ProvideReducedEvents.req(self)

        if not self.pilot:
            if self.producer_insts:
                reqs["producers"] = [
                    self.reqs.ProduceColumns.req(self, producer=producer_inst.cls_name)
                    for producer_inst in self.producer_insts
                    if producer_inst.produced_columns
                ]

            # add weight_producer dependent requirements
            reqs["weight_producer"] = law.util.make_unique(law.util.flatten(self.weight_producer_inst.run_requires()))

        return reqs

    def requires(self):
        reqs = {"events": self.reqs.ProvideReducedEvents.req(self)}

        if self.producer_insts:
            reqs["producers"] = [
                self.reqs.ProduceColumns.req(self, producer=producer_inst.cls_name)
                for producer_inst in self.producer_insts
                if producer_inst.produced_columns
            ]

        # add weight_producer dependent requirements
        reqs["weight_producer"] = law.util.make_unique(law.util.flatten(self.weight_producer_inst.run_requires()))

        return reqs

    workflow_condition = ReducedEventsUser.workflow_condition.copy()

    @workflow_condition.output
    def output(self):
        return  {"hists": self.target(f"ff_hist_{self.branch}.pickle")}
    @law.decorator.notify
    @law.decorator.log
    @law.decorator.localize(input=True, output=False)
    @law.decorator.safe_output
    def run(self):
        import hist
        import numpy as np
        import awkward as ak
        from columnflow.columnar_util import (
            Route, update_ak_array, add_ak_aliases, has_ak_column, attach_coffea_behavior, EMPTY_FLOAT
        )
        from columnflow.hist_util import fill_hist
        # prepare inputs
        inputs = self.input()

        # declare output: dict of histograms
        histograms = {}

        # run the weight_producer setup
        producer_reqs = self.weight_producer_inst.run_requires()
        reader_targets = self.weight_producer_inst.run_setup(producer_reqs, luigi.task.getpaths(producer_reqs))

        # create a temp dir for saving intermediate files
        tmp_dir = law.LocalDirectoryTarget(is_tmp=True)
        tmp_dir.touch()

        # get shift dependent aliases
        aliases = self.local_shift_inst.x("column_aliases", {})
        ff_variables = [var.var_route for var in self.config_inst.x.fake_factor_method.axes.values()]
        # define columns that need to be read
        
        read_columns = {Route("process_id")}
        read_columns |= set(map(Route, self.category_id_columns))
        read_columns |= set(self.weight_producer_inst.used_columns)
        read_columns |= set(map(Route, aliases.values()))
        read_columns |= set(map(Route, ff_variables))
        # empty float array to use when input files have no entries
        empty_f32 = ak.Array(np.array([], dtype=np.float32))

        # iterate over chunks of events and diffs
        file_targets = [inputs["events"]["events"]]
        if self.producer_insts:
            file_targets.extend([inp["columns"] for inp in inputs["producers"]])
            
        # prepare inputs for localization
        with law.localize_file_targets(
            [*file_targets, *reader_targets.values()],
            mode="r",
        ) as inps:
            
            for (events, *columns), pos in self.iter_chunked_io(
                [inp.abspath for inp in inps],
                source_type=len(file_targets) * ["awkward_parquet"] + [None] * len(reader_targets),
                read_columns=(len(file_targets) + len(reader_targets)) * [read_columns],
                chunk_size=self.weight_producer_inst.get_min_chunk_size(),
            ):
                # optional check for overlapping inputs
                if self.check_overlapping_inputs:
                    self.raise_if_overlapping([events] + list(columns))
                # add additional columns
                events = update_ak_array(events, *columns)
                # add aliases
                events = add_ak_aliases(
                    events,
                    aliases,
                    remove_src=True,
                    missing_strategy=self.missing_column_alias_strategy,
                )

                # attach coffea behavior aiding functional variable expressions
                events = attach_coffea_behavior(events)
                # build the full event weight
                if hasattr(self.weight_producer_inst, "skip_func") and not self.weight_producer_inst.skip_func():
                    events, weight = self.weight_producer_inst(events)
                else:
                    weight = ak.Array(np.ones(len(events), dtype=np.float32))
                # define and fill histograms, taking into account multiple axes
                category_ids = ak.concatenate(
                        [Route(c).apply(events) for c in self.category_id_columns],
                        axis=-1,)
                sr_names = self.categories
                for sr_name in sr_names:
                    the_sr = self.config_inst.get_category(sr_name)
                    regions = [sr_name]
                    if the_sr.aux:
                        for the_key in the_sr.aux.keys():
                            if (the_key == 'abcd_regs') or (the_key == 'ff_regs'):
                                regions += list(the_sr.aux[the_key].values())
                    else:
                        raise KeyError(f"Application and determination regions are not found for {the_sr}. \n Check aux field of the category map!") 
    
                    for region in regions: 
                        #by accessing the list of categories we check if the category with this name exists
                        cat = self.config_inst.get_category(region)
                        
                        # get variable instances
                        mask = ak.any(category_ids == cat.id, axis = 1)
                        masked_events = events[mask]
                        masked_weight = weight[mask]
                        
                        h = (hist.Hist.new.IntCat([], name="process", growth=True))
                        for (var_name, var_axis) in self.config_inst.x.fake_factor_method.axes.items(): 
                            h = eval(f'h.{var_axis.ax_str}') 
                        
                        h = h.Weight()
                        # broadcast arrays so that each event can be filled for all its categories
                        
                        fill_data = {
                            "process": masked_events.process_id,
                            "weight"  : masked_weight,
                        }
                        for (var_name, var_axis) in self.config_inst.x.fake_factor_method.axes.items(): 
                            route = Route(var_axis.var_route)
                            if len(masked_events) == 0 and not has_ak_column(masked_events, route):
                                values = empty_f32
                            else:
                                values = route.apply(masked_events)
                                if values.ndim != 1: values = ak.firsts(values,axis=1)
                                values = ak.fill_none(values, EMPTY_FLOAT)
                                
                                if var_name == 'n_jets': values = ak.where (values > 2, 
                                                                            2 * ak.ones_like(values),
                                                                            values) 
                                
                                if 'Int' in var_axis.ax_str: values = ak.values_astype(values, np.int64)
                            fill_data[var_name] = values
                        # fill it
                        fill_hist(
                            h,
                            fill_data,
                        )
                        if cat.name not in histograms.keys():
                            histograms[cat.name] = h
                        else:
                            histograms[cat.name] +=h
                        
        # merge output files
        self.output()["hists"].dump(histograms, formatter="pickle")
    
   


# overwrite class defaults
check_overlap_tasks = law.config.get_expanded("analysis", "check_overlapping_inputs", [], split_csv=True)
PrepareFakeFactorHistograms.check_overlapping_inputs = ChunkedIOMixin.check_overlapping_inputs.copy(
    default=PrepareFakeFactorHistograms.task_family in check_overlap_tasks,
    add_default_to_description=True,
)


PrepareFakeFactorHistogramsWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=PrepareFakeFactorHistograms,
    enable=["configs", "skip_configs", "datasets", "skip_datasets", "shifts", "skip_shifts"],
)


class MergeFakeFactorHistograms(
    #VariablesMixin,
    #WeightProducerMixin,
    #MLModelsMixin,
    #ProducersMixin,
    #SelectorStepsMixin,
    #CalibratorsMixin,
    DatasetTask,
    law.LocalWorkflow,
    RemoteWorkflow,
):
    only_missing = luigi.BoolParameter(
        default=False,
        description="when True, identify missing variables first and only require histograms of "
        "missing ones; default: False",
    )
    remove_previous = luigi.BoolParameter(
        default=False,
        significant=False,
        description="when True, remove particlar input histograms after merging; default: False",
    )

    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        PrepareFakeFactorHistograms=PrepareFakeFactorHistograms,
    )

    @classmethod
    def req_params(cls, inst: AnalysisTask, **kwargs) -> dict:
        _prefer_cli = law.util.make_set(kwargs.get("_prefer_cli", [])) | {"variables"}
        kwargs["_prefer_cli"] = _prefer_cli
        return super().req_params(inst, **kwargs)

    def create_branch_map(self):
        # create a dummy branch map so that this task could be submitted as a job
        return {0: None}

    # def _get_variables(self):
    #     if self.is_workflow():
    #         return self.as_branch()._get_variables()

    #     variables = self.variables

    #     # optional dynamic behavior: determine not yet created variables and require only those
    #     if self.only_missing:
    #         missing = self.output().count(existing=False, keys=True)[1]
    #         variables = sorted(missing, key=variables.index)

    #     return variables

    def workflow_requires(self):
        reqs = super().workflow_requires()

        if not self.pilot:
            #variables = self._get_variables()
            #if variables:
            reqs["hists"] = self.reqs.PrepareFakeFactorHistograms.req_different_branching(
                    self,
                    branch=-1,
                    #variables=tuple(variables),
            )

        return reqs

    def requires(self):
        #variables = self._get_variables()
        #if not variables:
        #    return []

        return self.reqs.PrepareFakeFactorHistograms.req_different_branching(
            self,
            branch=-1,
            #variables=tuple(variables),
            workflow="local",
        )

    def output(self):
        return {"hists": self.target(f"merged_ff_hist.pickle")}

    @law.decorator.notify
    @law.decorator.log
    def run(self):
        # preare inputs and outputs
        inputs = self.input()["collection"]
        outputs = self.output()

        # load input histograms
        hists = [
            inp["hists"].load(formatter="pickle")
            for inp in self.iter_progress(inputs.targets.values(), len(inputs), reach=(0, 50))
        ]
        cats = list(hists[0].keys())
        get_hists = lambda hists, cat : [h[cat] for h in hists]
        # create a separate file per output variable
        merged_hists = {}
        self.publish_message(f"merging {len(hists)} histograms for {self.dataset}")
        for the_cat in cats:
            h = get_hists(hists, the_cat)
            merged_hists[the_cat] = sum(h[1:], h[0].copy())
        outputs["hists"].dump(merged_hists, formatter="pickle")
        # optionally remove inputs
        if self.remove_previous:
            inputs.remove()

MergeFakeFactorHistogramsWrapper = wrapper_factory(
    base_cls=AnalysisTask,
    require_cls=MergeFakeFactorHistograms,
    enable=["configs", "skip_configs", "datasets", "skip_datasets", "shifts", "skip_shifts"],
)

class ComputeFakeFactors(
    DatasetsProcessesMixin,
    CategoriesMixin,
    WeightProducerMixin,
    ProducersMixin,
):
    sandbox = dev_sandbox(law.config.get("analysis", "default_columnar_sandbox"))

    only_missing = luigi.BoolParameter(
        default=False,
        description="when True, identify missing variables first and only require histograms of "
        "missing ones; default: False",
    )
    remove_previous = luigi.BoolParameter(
        default=False,
        significant=False,
        description="when True, remove particlar input histograms after merging; default: False",
    )
    
    # upstream requirements
    reqs = Requirements(
        RemoteWorkflow.reqs,
        MergeFakeFactorHistograms=MergeFakeFactorHistograms,
    )
    
    def store_parts(self):
        parts = super().store_parts()
        parts.insert_before("version", "datasets", f"datasets_{self.datasets_repr}")
        return parts
    
    @classmethod
    def req_params(cls, inst: AnalysisTask, **kwargs) -> dict:
        _prefer_cli = law.util.make_set(kwargs.get("_prefer_cli", [])) | {"variables"}
        kwargs["_prefer_cli"] = _prefer_cli
        return super().req_params(inst, **kwargs)
    
    def create_branch_map(self):
        # create a dummy branch map so that this task could be submitted as a job
        return {0: None}

        return reqs
    def requires(self):
        return {
            d: self.reqs.MergeFakeFactorHistograms.req_different_branching(
                self,
                branch=-1,
                dataset=d,
                workflow="local",
            )
            for d in self.datasets
        }
    
    def output(self):
        year = self.config_inst.campaign.aux['year']
        tag = self.config_inst.campaign.aux['tag']
        channel = self.config_inst.channels.get_first().name
        return {
            "ff_json": self.target('_'.join(('fake_factors', channel, str(year), tag)) + '.json'),
            "plots": {
                f"qcd_{s}_N_b_jets_{nb}": self.target(f"fake_factor_qcd_{s}_Nbjets_{nb}.png")
                for s in ['nominal', 'up', 'down']
                for nb in [0, 1, 2]
            },
            "plots1d": {
                f"qcd_{nj}_{nb}": self.target(f"fake_factor_qcd_Njets_{nj}_Nbjets_{nb}.png")
                for nj in [0, 1, 2, 3]
                for nb in [0, 1, 2]
            },
            "fitres": self.target('_'.join(('fitres', channel, str(year), tag)) + '.json'),
        }

    @law.decorator.log
    def run(self):
        import hist, numpy as np, matplotlib.pyplot as plt
        from scipy.optimize import curve_fit
        import correctionlib.schemav2 as cs

        # Plot style
        plt.figure(dpi=200)
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "monospace",
            "font.monospace": 'Computer Modern Typewriter'
        })

        # Load and merge histograms
        merged = {}
        for ds in self.input().values():
            for f in ds['collection'][0].values():
                for cat, h in f.load(formatter='pickle').items():
                    merged.setdefault(cat, []).append(h)

        # Separate MC and data
        mc_hists, data_hists = {}, {}
        for cat, hs in merged.items():
            for h in hs:
                for name in self.config_inst.processes.names():
                    proc = self.config_inst.processes.get(name)
                    if proc.id in h.axes['process']:
                        sel = h[{ 'process': hist.loc(proc.id) }].copy()
                        if proc.is_mc and not proc.has_tag('signal'):
                            mc_hists[cat] = mc_hists.get(cat, 0) + sel
                        elif proc.is_data:
                            data_hists[cat] = data_hists.get(cat, 0) + sel

        def eval_formula(fstr, popt, roundit=False):
            for i, p in enumerate(popt):
                rep = f"{p:.3e}" if roundit else str(p)
                fstr = fstr.replace(f'p{i}', rep)
            return fstr

        def get_ff_corr(data, mc, num_reg, den_reg, name, label):
            # Extract numerator/denominator hist
            def get_cat(h, reg):
                key = self.config_inst.get_category(self.categories[0]).aux['ff_regs'][reg]
                return h[key]

            dnum = get_cat(data, num_reg)
            dden = get_cat(data, den_reg)
            mnum = get_cat(mc, num_reg)
            mden = get_cat(mc, den_reg)

            num = dnum.values() - mnum.values()
            den = dden.values() - mden.values()
            # from IPython import embed; embed()
            ff_vals = np.where((num > 0) & (den > 0), num / np.maximum(den, 1), -1)
            ff_err = ff_vals * ((np.sqrt(dnum.variances()) + np.sqrt(mnum.variances())) / np.abs(num)
                                + (np.sqrt(dden.variances()) + np.sqrt(mden.variances())) / np.abs(den))
            ff_err[ff_vals < 0] = 1

            # Build raw histogram
            hbase = hist.Hist.new
            for ax in self.config_inst.x.fake_factor_method.axes.values():
                hbase = eval(f'hbase.{ax.ax_str}')
            hbase = hbase.StrCategory(['nominal', 'up', 'down'], name='syst', label='Stat Unc')
            raw = hbase.Weight()
            raw.view().value[..., 0] = ff_vals
            raw.view().variance[..., 0] = ff_err ** 2
            raw.name = name + '_raw'
            raw.label = label

            # Prepare fitted histogram
            fit_hist = raw.copy().reset()
            fit_hist.name = name
            fit_hist.label = label

            fit_results = {}
            for nb in raw.axes['N_b_jets']:
                fit_results[nb] = {}
                for nj in raw.axes['N_jets']:
                    slice1d = raw[{ 'N_b_jets': hist.loc(nb), 'N_jets': hist.loc(nj), 'syst': hist.loc('nominal') }]
                    x = slice1d.axes[0].centers
                    y = slice1d.values()
                    yerr = np.sqrt(slice1d.variances())
                    mask = y > 0

                    # Choose fit function
                    # if nj == 0:
                    func = lambda xx, p0, p1, p2: p0 + p1 * xx + p2 * xx ** 2
                    fstr = 'p0+p1*x+p2*x*x'
                        # bounds = ([-10, -5, -1], [10, 5, 1])
                    # else:
                    #     func = lambda xx, p0, p1, p2: p0 + p1 * np.exp(-p2 * xx)
                    #     fstr = 'p0+p1*exp(-p2*x)'
                    #     # bounds = ([-0.5, -3, 0], [0.5, 3, 0.1])

                    if mask.sum() >= 3:
                        popt, pcov = curve_fit(func, x[mask], y[mask], sigma=yerr[mask], maxfev=5000, absolute_sigma=True)
                    else:
                        popt = np.zeros(3)
                        pcov = np.zeros((3, 3))

                    # Numeric fit closure
                    fit_func = lambda xx, f=func, p=popt: f(xx, *p)
                    fit_results[nb][nj] = {'popt': popt, 'pcov': pcov, 'fstr': fstr, 'func': fit_func}

                    # Fill fitted histogram
                    for i, shift in enumerate(['down', 'nominal', 'up']):
                        vals = func(x, *(popt + (i - 1) * np.sqrt(np.diag(pcov))))
                        fit_hist.view().value[:, fit_hist.axes['N_jets'].index(nj),
                                                fit_hist.axes['N_b_jets'].index(nb),
                                                fit_hist.axes['syst'].index(shift)] = vals

            return raw, fit_hist, fit_results

        # Compute QCD fake factors
        q_raw, q_fit, q_res = get_ff_corr(data_hists, mc_hists, 'dr_num_qcd', 'dr_den_qcd', 'ff_qcd', 'Fake QCD')

        # Build and dump correction set
        corr = cs.Correction(
            name='ff_qcd', description='Fake factor QCD', version=2,
            inputs=[
                cs.Variable(name='delta_r', type='real', description='Delta R'),
                cs.Variable(name='N_jets', type='int', description='Number of jets'),
                cs.Variable(name='N_b_jets', type='int', description='Number of b jets')
            ],
            output=cs.Variable(name='weight', type='real', description='Weight'),
            data=cs.Category(
                nodetype='category', input='N_b_jets', content=[
                    cs.CategoryItem(
                        key=nb,
                        value=cs.Category(
                            nodetype='category', input='N_jets', content=[
                                cs.CategoryItem(
                                    key=nj,
                                    value=cs.Formula(
                                        nodetype='formula', variables=['delta_r'], parser='TFormula',
                                        expression=eval_formula(q_res[nb][nj]['fstr'], q_res[nb][nj]['popt'])
                                    )
                                ) for nj in q_res[nb]
                            ]
                        )
                    ) for nb in q_res
                ]
            )
        )
        cset = cs.CorrectionSet(schema_version=2, description='Fake factors', corrections=[corr])
        self.output()['ff_json'].dump(cset.json(exclude_unset=True), formatter='json')
        self.output()['fitres'].dump(str(q_res), formatter='json')

        # Plotting
        for nb in q_raw.axes['N_b_jets']:
            fig, ax = plt.subplots(figsize=(12, 8))
            h2d = q_raw[{ 'N_b_jets': hist.loc(nb), 'syst': hist.loc('nominal') }]
            pcm = ax.pcolormesh(*np.meshgrid(*h2d.axes.edges), h2d.view().value.T)
            ax.set_xlabel(h2d.axes[0].label)
            ax.set_ylabel(h2d.axes[1].label)
            plt.colorbar(pcm, ax=ax)
            self.output()['plots'][f'qcd_nominal_N_b_jets_{nb}'].dump(fig, formatter='mpl')
            for nj in q_raw.axes['N_jets']:
                fig, ax = plt.subplots(figsize=(8, 6))
                h1d = q_raw[{ 'N_jets': hist.loc(nj), 'N_b_jets': hist.loc(nb), 'syst': hist.loc('nominal') }]
                x, y = h1d.axes[0].centers, h1d.counts()
                yerr = np.sqrt(h1d.variances()).flatten()
                ax.errorbar(x, y, yerr=yerr, fmt='o', capsize=4)
                func = q_res[nb][nj]['func']
                xf = np.linspace(x.min(), x.max(), 100)
                ax.plot(xf, func(xf))
                ax.set_xlabel('Delta R')
                ax.set_ylabel('Fake Factor')
                self.output()['plots1d'][f'qcd_{nj}_{nb}'].dump(fig, formatter='mpl')
