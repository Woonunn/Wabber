"""Live LangGraph Capture and decision Replay, using real ATSuite Tool RPC."""
import argparse,asyncio,json,traceback,uuid,time
from pathlib import Path
from atsuite.tracing import TraceCollector
from atsuite.tracing.langgraph import LangGraphTracingAdapter
from dotenv import load_dotenv
from benchmarks.DataSciBench.langgraph_agent.graph import build_graph
from benchmarks.DataSciBench.langgraph_agent.model_client import DeepSeekClient
from benchmarks.DataSciBench.langgraph_agent.tool_client import ToolClient
from benchmarks.DataSciBench.langgraph_agent.trace_recorder import TraceRecorder
from benchmarks.DataSciBench.langgraph_agent.trace_replayer import ReplayModel

ROOT=Path(__file__).resolve().parents[3]


async def run(task_id,config,output,capture=None,workload_model=None,provider_model=None,provider_context=None,*,calibration=False,router=None,routing_context=None,prompt=None,manifest=None,arrival_ns=None,workload="DataSciBench"):
    if workload not in {'DataSciBench','TravelPlanner','ClaudeCodeReview','ScientificComputation'}:
        raise ValueError('Unknown LangGraph workload')
    if workload != 'DataSciBench' and manifest is None:
        raise ValueError('A workload task manifest is required')
    if workload != 'DataSciBench' and (calibration or prompt is not None):
        raise ValueError('E3 uses sealed prompts and strict business-result Replay')
    if router is not None:
        if workload_model is not None or provider_model is not None:
            raise ValueError('Router owns both model completion hooks')
        if not routing_context or set(routing_context['targets']) != set(router.pools):
            raise ValueError('Routing requires one audited target per pool')
        if routing_context.get('shadow'):
            actual=routing_context.get('actual_pool')
            if actual not in router.pools or routing_context['targets'][actual]['tool'] != config['tool']:
                raise ValueError('Shadow actual pool must match the unchanged Tool configuration')
        for key,target in routing_context['targets'].items():
            if target['tool']['provider'] != router.pools[key]['provider']:
                raise ValueError('Routing target/provider mismatch')
    if prompt is not None and (not isinstance(prompt,str) or not prompt.strip() or capture):
        raise ValueError('A new Prompt requires live planning and nonempty text')
    if calibration and not capture:raise ValueError("Calibration requires a frozen Capture")
    if provider_model is not None:
        from atsuite.provider_model.observation import validate_pool,features
        if provider_context is None:raise ValueError('Provider modeling requires pool and load context')
        if provider_context.get('reference_profile') is not None and not provider_context['reference_profile'].get('reference_id'):
            raise ValueError('An explicit reference profile requires its shared reference_id')
        validate_pool(provider_context['pool'])
        if provider_context['pool']['provider']!=config['tool']['provider']:raise ValueError('Provider context does not match Tool backend')
        features(provider_context.get('reference_profile',{}),provider_context['pool'],provider_context['load_rps'])
    tasks=json.loads(Path(manifest or ROOT/'benchmarks/DataSciBench/tasks/manifest.json').read_text())
    task=next(t for t in tasks['tasks'] if t['task_id']==task_id)
    for instruction in ('planner_instruction','validation_contract'):
        if instruction in config:
            if not isinstance(config[instruction],str) or not config[instruction].strip():
                raise ValueError('Agent instructions must be nonempty strings')
            task={**task,instruction:config[instruction]}
    if prompt is not None:
        import hashlib
        task={**task,'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
    from atsuite.workload import describe
    descriptor=describe(prompt=task['prompt'],files=task['readiness']['inputs']['files'],
                        tools=['prepare_workspace','execute_analysis','merge_artifacts','validate_result'] if workload=='DataSciBench' else [d['name'] for d in task['tool_definitions']],
                        category=workload)
    session_id=uuid.uuid4().hex;request_id=uuid.uuid4().hex
    if router is not None:request_id=routing_context.get('request_id',request_id)
    tracing=LangGraphTracingAdapter(TraceCollector(request_id,session_id,output))
    recorder=TraceRecorder(output,request_id,session_id,tracing)
    initial={'request_id':request_id,'session_id':session_id,'task_id':task_id,'user_prompt':task['prompt'],
         'messages':[],'workspace_id':'','dataset_refs':[{k:f[k] for k in ('name','sha256','bytes')} for f in task['readiness']['inputs']['files']],
         'execution_plan':[],'tool_results':{},'artifacts':[],'iteration':0,'status':'new','error':None,'completed':[],'snapshot':0}
    arrival_ns=time.time_ns() if arrival_ns is None else arrival_ns
    if router is not None:arrival_ns=routing_context.get('arrival_ns',arrival_ns)
    if workload_model is not None:
        from atsuite.tracing.storage import write_json
        prediction=workload_model.begin(request_id,session_id,arrival_ns,descriptor=descriptor)
        write_json(output/'workload_prediction.json',prediction)
    if provider_model is not None:
        from atsuite.tracing.storage import write_json
        reference_workload=provider_context.get('reference_profile')
        if reference_workload is None:reference_workload=prediction if workload_model is not None else []
        write_json(output/'provider_prediction.json',provider_model.predict(
            reference_workload,[provider_context['pool']],provider_context['load_rps'],as_of_ns=arrival_ns))
    tools=model=None;state=initial;sessions_closed=False
    try:
        if router is not None:
            decision=router.decide(request_id,None,routing_context['load_rps'],as_of_ns=time.time_ns(),arrival_ns=arrival_ns,
                                   shadow=routing_context.get('shadow',False),
                                   pool_load_rps=routing_context.get('pool_load_rps'),descriptor=descriptor,
                                   allowed_pools=routing_context.get('allowed_pools'),
                                   **({'fallback_pool':routing_context['fallback_pool']} if routing_context.get('fallback_pool') else {}))
            from atsuite.tracing.storage import write_json
            write_json(output/'routing_decision.json',decision)
            if not decision['shadow']:
                if decision['selected'] is None:raise RuntimeError('No trusted feasible execution pool')
                target=routing_context['targets'][decision['selected']]
                config={**config,'tool':target['tool']}
                capture=target.get('capture',capture)
        recorder.emit('run_start',workload=workload,mode='environment_reference' if calibration else 'replay' if capture else 'capture',task_id=task_id,prompt_sha256=task['prompt_sha256'],
             config=config,source_capture=str(capture) if capture else None,execution_scope=config['tool']['execution_scope'])
        from experiments.e3.agent import result_view
        equality={}
        if workload=='ScientificComputation':
            from benchmarks.ScientificComputation.langgraph_agent.graph import numeric_equal
            equality={'result_equal':numeric_equal}
        tools=await asyncio.to_thread(ToolClient,config['tool'],recorder,capture,calibration,
                                      **({'result_view':result_view,**equality} if workload!='DataSciBench' else {}))
        model=ReplayModel(capture,recorder,tools.comparable,calibration,**({'input_equal':numeric_equal} if workload=='ScientificComputation' else {})) if capture else DeepSeekClient(recorder,config['model'])
        if workload!='DataSciBench' and not capture:model.tool_definitions=task['tool_definitions']
        if workload=='DataSciBench':
            graph=build_graph(model,tools,recorder,task['output_files'],config['max_iterations'],config['model']['seed'],task.get('validation_contract'),task.get('task_type')=='bcb')
        else:
            import importlib
            builder=importlib.import_module(f'benchmarks.{workload}.langgraph_agent.graph').build_graph
            graph=builder(model,tools,task,config['max_iterations'])
        timeout=config.get('session_timeout_s',900)
        if router is not None and decision.get('execution_bounds'):
            timeout=min(timeout,decision['execution_bounds']['timeout_s']-(time.time_ns()-arrival_ns)/1e9)
        async with asyncio.timeout(max(0,timeout)):
            async for current in tracing.astream(graph,initial,config={'recursion_limit':150,'max_concurrency':config['max_parallel']}):
                state=current
        if state['status']!='complete':raise RuntimeError('Agent did not complete: '+state.get('feedback',''))
        if capture:model.assert_consumed();tools.assert_consumed()
        return {'status':'complete','directory':str(output),'task_id':task_id,'iterations':state['iteration']+1}
    except BaseException as exc:
        recorder.emit('run_error',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        state={**state,'status':'failed','error':{'type':type(exc).__name__,'message':str(exc)}}
        raise
    finally:
        try:
            if tools:
                await tools.aclose()
                sessions_closed=True
        except BaseException as exc:
            state={**state,'status':'failed','error':{'type':type(exc).__name__,'message':str(exc)}}
            recorder.emit('run_error',error_type=type(exc).__name__,error=str(exc),phase='session_close')
            if router is not None and hasattr(router,'hold_release'):
                router.hold_release(request_id)
            raise
        finally:
            recorder.finish(state)
            from atsuite.tracing.storage import write_json
            from atsuite.workload import extract_profile
            trace=tracing.collector.snapshot()
            trace.update(arrival_ns=arrival_ns,completed_ns=time.time_ns(),
                         sessions_closed=sessions_closed,
                         request_status='success' if state['status']=='complete' else 'failed')
            profile=workload_model.observe(trace,descriptor=descriptor) if workload_model is not None else extract_profile(trace)
            write_json(output/'workload_profile.json',profile)
            if provider_model is not None:
                observation=provider_model.complete(trace,provider_context['pool'],provider_context['load_rps'],
                    reference_profile=provider_context.get('reference_profile'))
                write_json(output/'provider_observation.json',observation)
            if router is not None and (request_id in router.active or
                    router.decisions.get(request_id,{}).get('shadow')):
                observation=router.complete(request_id,trace,actual_pool=routing_context.get('actual_pool')
                                            if routing_context.get('shadow') else None)
                write_json(output/'provider_observation.json',observation)
            if model and not capture:await model.client.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',required=True)
    p.add_argument('--config',type=Path,default=ROOT/'benchmarks/DataSciBench/config/langgraph_experiment.json')
    p.add_argument('--output',type=Path);p.add_argument('--replay',type=Path)
    p.add_argument('--workload',choices=['DataSciBench','TravelPlanner','ClaudeCodeReview','ScientificComputation'],default='DataSciBench')
    p.add_argument('--manifest',type=Path,help='Independent prepared task manifest (default: six-task manifest)')
    p.add_argument('--workload-model',type=Path,help='Load/update an empirical workload snapshot')
    p.add_argument('--provider-model',type=Path,help='Load/update provider observations at request completion')
    p.add_argument('--provider-context',type=Path,help='JSON pool, load_rps and optional fixed reference_profile')
    a=p.parse_args();load_dotenv(ROOT/'.env',override=False)
    if bool(a.provider_model)!=bool(a.provider_context):p.error('--provider-model and --provider-context are required together')
    config=json.loads(a.config.read_text());output=a.output or ROOT/'results/phase2'/uuid.uuid4().hex
    from atsuite.workload import WorkloadModel
    from atsuite.tracing.storage import trace_path
    model=(WorkloadModel.load(a.workload_model) if trace_path(a.workload_model).exists() else WorkloadModel()) if a.workload_model else None
    from atsuite.provider_model import ProviderModel
    provider_model=(ProviderModel.load(a.provider_model) if trace_path(a.provider_model).exists() else ProviderModel()) if a.provider_model else None
    provider_context=json.loads(a.provider_context.read_text()) if a.provider_context else None
    try:
        print(json.dumps(asyncio.run(run(a.task,config,output,a.replay,model,provider_model,provider_context,manifest=a.manifest,workload=a.workload)),ensure_ascii=False))
    finally:
        if model is not None:model.save(a.workload_model)
        if provider_model is not None:provider_model.save(a.provider_model)


if __name__=='__main__':main()
