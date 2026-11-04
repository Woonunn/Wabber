"""Parallel, isolated ATSuite builds/deployments; read back the actual platform settings."""
import argparse
import asyncio
import json
import hashlib
import io
import tarfile
import uuid
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
from dotenv import load_dotenv
from experiments.exp1.data.prepare_e1a import ROOT,CONFIG,sha
from atsuite.pipeline import resolve_benchmark,target_image_tag
from atsuite.utils import normalize_node_name,state_bucket_for_provider
from atsuite.tracing.storage import trace_path


def make_configs(config):
    base=json.loads((ROOT/'benchmarks/DataSciBench/config/langgraph_atg.json').read_text())
    if config.get('workload'):
        from experiments.exp1.data.prepare_e1b import load
        sealed,_=load(ROOT/config['workload'])
        selected=sealed['tasks'][config['task_id']]
    else:
        captures=json.loads((ROOT/'benchmarks/DataSciBench/trace/captured/manifest.json').read_text())['captures']
        selected=next(t for t in captures if t['task_id']==config['task_id'])
    paths={};experiment=config['experiment'].lower().replace('-','')
    for provider in config['providers']:
        value=json.loads(json.dumps(base));name=experiment.capitalize()+''.join(p.title() for p in provider.split('_'))
        resources=config.get('provider_resources',{}).get(provider,config['resources'])
        if provider=='aws_lambda':value['pipeline']['faas']['build']['python_version']='3.12'
        value['trace_file']='./trace/captured/'+Path(selected['capture']).name+'/trace.atg.json'
        if config.get('task_manifest'):
            for family in ('faas','mcp_serverless'):value['pipeline'][family]['build']['python_version']='3.12.12'
            if provider.startswith('ali_'):
                # Ali's published base has no Python 3.12; FC custom containers support the pinned standard image.
                for family in ('faas','mcp_serverless'):
                    value['pipeline'][family]['build']['base_image']='python:3.12.12-slim@sha256:f3fa41d74a768c2fce8016b98c191ae8c1bacd8f1152870a3f9f87d350920b7c'
            value['trace_file']=str((ROOT/config['workload']/'trace.atg.json').resolve())
        if config.get('workload'):value['experiment_workload']=str((ROOT/config['workload']).resolve())
        unit=value['pipeline']['faas']['units'][0];unit['name']=name;unit['deploy']=resources
        server=value['pipeline']['mcp_serverless']['servers'][0];server['name']=name;server['deploy']=resources
        path=ROOT/'benchmarks/DataSciBench/config'/(experiment+'_'+provider+'.json')
        path.write_text(json.dumps(value,indent=2)+'\n');resolve_benchmark.cache_clear();resolve_benchmark(path,provider)
        paths[provider]=path
    return paths



def audit_image(provider, config_path):
    resolved=resolve_benchmark(config_path,provider);target=next(iter(resolved.targets.values()))
    image=target_image_tag(resolved.bench_name,target)
    metadata=json.loads(subprocess.check_output(['docker','image','inspect',image],text=True))[0]
    name='wabber-e1a-audit-'+uuid.uuid4().hex
    subprocess.run(['docker','create','--name',name,'--entrypoint','/bin/true',metadata['Id']],check=True,stdout=subprocess.DEVNULL)
    prefix='/app/mcp/AgentTools/' if target.family=='session' else '/app/'
    files={};approved=None
    try:
        expected={prefix+f:ROOT/'benchmarks/DataSciBench/nodes/AgentTools'/f for f in ('implementation.py','native_kernel.py','guards.py','sandbox_worker.py','logical_state.py','frozen_policy.py')}
        expected[prefix+'kernel.py']=ROOT/'benchmarks/DataSciBench/nodes/DataSci/kernel.py'
        expected.update({'/app/atsuite_sdk/'+f:ROOT/'atsuite_sdk'/f for f in ('state.py','response.py','function.py','mcp_server.py')})
        if provider.startswith('ali_'):expected['/app/atsuite_sdk/oss.py']=ROOT/'atsuite/ali/oss.py'
        for destination,source in expected.items():
            raw=subprocess.check_output(['docker','cp',name+':'+destination,'-'])
            with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
                member=next(m for m in archive.getmembers() if m.isfile());actual=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            if actual!=sha(source):raise ValueError('Image contains stale source: '+destination)
            files[destination]=actual
        raw=subprocess.check_output(['docker','cp',name+':'+prefix+'approved_trace.json','-'])
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            member=next(m for m in archive.getmembers() if m.isfile());approved_raw=archive.extractfile(member).read();approved=json.loads(approved_raw)
        policies=approved.get('tasks') or {approved['task_id']:approved}
        workload=json.loads(Path(config_path).read_text()).get('experiment_workload')
        if workload:
            from experiments.exp1.data.prepare_e1b import load
            sealed,_=load(workload);captures=sealed['tasks']
            if set(policies)!=set(captures) or approved.get('workload_manifest_sha256')!=sha(Path(workload)/'manifest.json'):
                raise ValueError('Image must contain the complete sealed workload')
        else:captures={r['task_id']:r for r in json.loads((ROOT/'benchmarks/DataSciBench/trace/captured/manifest.json').read_text())['captures']}
        for task_id,policy in policies.items():
            if policy['task_id']!=task_id or policy['capture_events_sha256']!=sha(trace_path(ROOT/captures[task_id]['capture']/'events.jsonl')):raise ValueError('Image contains another Capture policy')
        if workload and sealed.get('task_manifest'):
            for filename in ('bcb_checks.py','result_checks.py','bcb_cases.json.gz'):
                source=ROOT/'benchmarks/DataSciBench/nodes/AgentTools'/filename
                if filename=='bcb_cases.json.gz':source=ROOT/'benchmarks/DataSciBench/tasks/mix/runtime_checks.json.gz'
                raw=subprocess.check_output(['docker','cp',name+':'+prefix+filename,'-'])
                with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
                    member=next(m for m in archive.getmembers() if m.isfile());actual=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                if actual!=sha(source):raise ValueError('Stale BCB validation source: '+filename)
                files[prefix+filename]=actual
        if 'tasks' not in approved:
            trace=json.loads(Path(config_path).read_text())['trace_file']
            capture=ROOT/'benchmarks/DataSciBench'/Path(trace).parent
            if approved['capture_events_sha256']!=sha(capture/'events.jsonl'):raise ValueError('Image contains another Capture policy')
        files[prefix+'approved_trace.json']=hashlib.sha256(approved_raw).hexdigest()
        for policy in policies.values():
            for ref in policy['dataset_refs']:
                raw=subprocess.check_output(['docker','cp',name+':'+prefix+'datasets/'+policy['task_id']+'/'+ref['name'],'-'])
                with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
                    member=next(m for m in archive.getmembers() if m.isfile());actual=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                if actual!=ref['sha256']:raise ValueError('Image dataset differs from the approved Capture')
    finally:subprocess.run(['docker','rm',name],check=True,stdout=subprocess.DEVNULL)
    result={'tag':image,'id':metadata['Id'],'size_bytes':metadata['Size'],'architecture':metadata['Architecture'],'os':metadata['Os'],'source_sha256':files}
    if 'tasks' in approved:
        result.update(approved_tasks={k:v['capture_events_sha256'] for k,v in policies.items()},workload_manifest_sha256=approved['workload_manifest_sha256'])
    else:result.update(approved_task_id=approved['task_id'],capture_events_sha256=approved['capture_events_sha256'])
    return result


def agentcore_usage_delivery(runtime_arn, runtime_id, region):
    import boto3
    logs=boto3.client('logs',region_name=region)
    group='/aws/vendedlogs/bedrock-agentcore/runtime/USAGE_LOGS/'+runtime_id
    try:logs.create_log_group(logGroupName=group)
    except logs.exceptions.ResourceAlreadyExistsException:pass
    logs.put_retention_policy(logGroupName=group,retentionInDays=7)
    account=runtime_arn.split(':')[4]
    group_arn=f'arn:aws:logs:{region}:{account}:log-group:{group}'
    name=runtime_id+'-e1a-usage'
    logs.put_delivery_source(name=name,logType='USAGE_LOGS',resourceArn=runtime_arn)
    destination=logs.put_delivery_destination(name=name,deliveryDestinationConfiguration={'destinationResourceArn':group_arn})['deliveryDestination']['arn']
    existing=[];token=None
    while True:
        page=logs.describe_deliveries(**({'nextToken':token} if token else {}));existing+=page.get('deliveries',[]);token=page.get('nextToken')
        if not token:break
    delivery=next((d for d in existing if d['deliverySourceName']==name and d['deliveryDestinationArn']==destination),None)
    if delivery is None:delivery=logs.create_delivery(deliverySourceName=name,deliveryDestinationArn=destination)['delivery']
    return {'log_group':group,'source':name,'destination_arn':destination,'delivery_id':delivery['id'],'retention_days':7}


def configure(provider, config_path, *, push_image=True, environment=None, concurrency_limit=64):
    if type(concurrency_limit) is not int or concurrency_limit < 1:
        raise ValueError('A positive integer concurrency limit is required')
    resolved=resolve_benchmark(config_path,provider);target=next(iter(resolved.targets.values()));name=normalize_node_name(target.name)
    bucket=state_bucket_for_provider(provider);common={'PROJECT':bucket,**(environment or {})}
    if provider=='aws_lambda':
        from atsuite.aws.aws import AWS
        api=AWS().get_lambda_client();current=api.get_function_configuration(FunctionName=name)
        api.update_function_configuration(FunctionName=name,Environment={'Variables':{**current.get('Environment',{}).get('Variables',{}),**common}})
        api.get_waiter('function_updated_v2').wait(FunctionName=name)
        api.put_function_concurrency(FunctionName=name,ReservedConcurrentExecutions=concurrency_limit)
        response=api.get_function_configuration(FunctionName=name)
        return {'region':api.meta.region_name,'runtime_name':name,'function_arn':response['FunctionArn'],
                'memory_mb':response['MemorySize'],'cpu_policy':'Lambda CPU scales with memory; requested CPU is not an independently enforceable setting',
                'ephemeral_storage':response.get('EphemeralStorage'),'timeout_s':response['Timeout'],'architectures':response.get('Architectures'),
                'reserved_concurrency':concurrency_limit,'instance_concurrency':1,'state_bucket':bucket,'warmup':'none','environment_keys':list(response.get('Environment',{}).get('Variables',{}))}
    if provider=='aws_agentcore':
        from atsuite.aws.aws import AWS
        api=AWS().get_agentcore_client();runtime_name='atsuite_'+name.lower().replace('-','_')
        runtimes=[];token=None
        while True:
            page=api.list_agent_runtimes(**({'nextToken':token} if token else {}));runtimes+=page.get('agentRuntimes',[]);token=page.get('nextToken')
            if not token:break
        entry=next(r for r in runtimes if r['agentRuntimeName']==runtime_name)
        current=api.get_agent_runtime(agentRuntimeId=entry['agentRuntimeId'])
        # CPU/memory are provider-managed in this API version; never claim that requested values were applied.
        kwargs={k:current[k] for k in ('agentRuntimeArtifact','roleArn','networkConfiguration','protocolConfiguration')}
        # Long Tools must finish before the platform can reclaim their resident Session.
        updated=api.update_agent_runtime(agentRuntimeId=entry['agentRuntimeId'],**kwargs,
            environmentVariables={**current.get('environmentVariables',{}),**common},
            lifecycleConfiguration={'idleRuntimeSessionTimeout':max(60,target.runtime.timeout+60),'maxLifetime':900})
        deadline=time.monotonic()+300
        while True:
            observed=api.get_agent_runtime(agentRuntimeId=entry['agentRuntimeId'])
            if observed['status']=='READY':break
            if observed['status']=='FAILED' or time.monotonic()>deadline:raise RuntimeError('AgentCore configuration did not become READY')
            time.sleep(2)
        return {'region':api.meta.region_name,'runtime_name':runtime_name,'runtime_id':entry['agentRuntimeId'],
                'runtime_arn':entry['agentRuntimeArn'],'resource_policy':'provider-managed; API exposes no CPU/memory setting',
                'cpu_requested':target.runtime.cpu,'memory_requested_mb':target.runtime.memory,
                'cpu_actual':None,'memory_actual_mb':None,'instance_concurrency':'platform-managed; measured in Pilot',
                'usage_delivery':agentcore_usage_delivery(entry['agentRuntimeArn'],entry['agentRuntimeId'],api.meta.region_name),'runtime_version':observed.get('agentRuntimeVersion'),'lifecycle_configuration':observed.get('lifecycleConfiguration'),'idle_timeout_s':observed['lifecycleConfiguration']['idleRuntimeSessionTimeout'],'max_lifetime_s':900,'state_bucket':bucket,'warmup':'none'}
    common['ATSUITE_OSS_INTERNAL']='true'
    from atsuite.ali.ali import Ali
    from alibabacloud_fc20230330 import models
    api=Ali().get_fc_client();function=name+('-function' if provider=='ali_fc' else '-mcp')
    current=api.get_function(function,models.GetFunctionRequest()).body
    image=target_image_tag(resolved.bench_name,target)
    metadata=json.loads(subprocess.check_output(['docker','image','inspect',image],text=True))[0]
    versioned=os.environ['DOCKERHUB_NAME']+':'+name.lower()+'-'+metadata['Id'].split(':')[1][:16]
    subprocess.run(['docker','tag',metadata['Id'],versioned],check=True)
    if push_image:subprocess.run(['docker','push',versioned],check=True)
    container=models.CustomContainerConfig(image=versioned,port=current.custom_container_config.port,entrypoint=current.custom_container_config.entrypoint)
    kwargs={'custom_container_config':container,'cpu':target.runtime.cpu,'memory_size':target.runtime.memory,'disk_size':target.runtime.disk,'timeout':target.runtime.timeout,'environment_variables':{**(current.environment_variables or {}),**common},'instance_concurrency':1 if provider=='ali_fc' else 200}
    if provider=='ali_agentrun':
        kwargs.update(session_affinity='MCP_STREAMABLE',instance_isolation_mode='SESSION_EXCLUSIVE',
            session_affinity_config=json.dumps({'sessionConcurrencyPerInstance':1,'SessionTTLInSeconds':900,'SessionIdleTimeoutInSeconds':60}))
    api.update_function(function,models.UpdateFunctionRequest(body=models.UpdateFunctionInput(**kwargs)))
    api.put_scaling_config(function,models.PutScalingConfigRequest(body=models.PutScalingConfigInput(min_instances=0)))
    api.put_concurrency_config(function,models.PutConcurrencyConfigRequest(body=models.PutConcurrencyInput(reserved_concurrency=concurrency_limit)))
    deadline=time.monotonic()+300
    while True:
        value=api.get_function(function,models.GetFunctionRequest()).body
        if value.last_update_status in {None,'Successful'} and value.state in {None,'Active'}:break
        if value.state=='Failed' or value.last_update_status=='Failed' or time.monotonic()>deadline:raise RuntimeError('Alibaba function update did not become Active: '+str(value.last_update_status_reason))
        time.sleep(2)
    return {'region':os.environ['ALI_REGION'],'runtime_name':function,'cpu':value.cpu,'memory_mb':value.memory_size,
            'disk_mb':value.disk_size,'timeout_s':value.timeout,'instance_concurrency':value.instance_concurrency,
            'session_affinity':value.session_affinity,'session_affinity_config':value.session_affinity_config,
            'instance_isolation_mode':value.instance_isolation_mode,'reserved_request_concurrency':concurrency_limit,'min_instances':0,
            'state_bucket':bucket,'state_transport':'same-region OSS internal endpoint','warmup':'none','image':value.custom_container_config.image,'resolved_image_uri':value.custom_container_config.resolved_image_uri,
            'note':'ali_agentrun uses the existing ATSuite FC session-exclusive MCP adapter; it is not a separate AgentRun API deployment'}


async def execute(config,output,*,deploy=False,providers=None,deploy_only=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False);paths=make_configs(config)
    env=dict(os.environ)
    env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','')
    if config.get('workload'):env['WABBER_BUILD_WORKLOAD']=str((ROOT/config['workload']).resolve())
    async def one(provider):
        path=paths[provider];start=time.time();record={'provider':provider,'config':str(path.relative_to(ROOT)),'config_sha256':sha(path),'start_utc_epoch':start}
        with (output/(provider+'.log')).open('w') as log:
            stages=[('build','atsuite.cli.build_docker_images',['--output',str(output.relative_to(ROOT)/'build'/provider)])]
            if deploy_only:stages=[]
            if deploy:stages.append(('deploy','atsuite.cli.deploy',[]))
            for stage,module,extra in stages:
                if stage=='deploy':
                    try:record['image']=audit_image(provider,path)
                    except Exception as exc:
                        record.update(status='failed',failed_stage='image_audit',error=str(exc));break
                for attempt in range(1,4):
                    process=await asyncio.create_subprocess_exec(sys.executable,'-m',module,'--config',str(path),'--provider',provider,*extra,
                        cwd=ROOT,env=env,stdout=log,stderr=asyncio.subprocess.STDOUT)
                    code=await process.wait()
                    record.setdefault('stage_attempts',[]).append({'stage':stage,'attempt':attempt,'exit_code':code})
                    if not code:break
                    if attempt<3:await asyncio.sleep(2)
                record[stage+'_exit_code']=code
                if code:
                    record.update(status='failed',failed_stage=stage);break
            else:
                record['status']='built'
                if not deploy:
                    try:record['image']=audit_image(provider,path)
                    except Exception as exc:record.update(status='failed',failed_stage='image_audit',error=str(exc))
                if deploy:
                    # Each readback subprocess owns its main thread and credential SDK.
                    observed_path=output/(provider+'-actual.json')
                    script="from dotenv import load_dotenv; from experiments.exp1.runner.deploy_e1a import configure,ROOT; import sys,json; from pathlib import Path; load_dotenv(ROOT/'.env'); Path(sys.argv[3]).write_text(json.dumps(configure(sys.argv[1],Path(sys.argv[2])),indent=2))"
                    for attempt in range(1,4):
                        process=await asyncio.create_subprocess_exec(sys.executable,'-c',script,provider,str(path),str(observed_path),cwd=ROOT,stdout=log,stderr=asyncio.subprocess.STDOUT)
                        code=await process.wait()
                        record.setdefault('stage_attempts',[]).append({'stage':'resource_readback','attempt':attempt,'exit_code':code})
                        if not code:break
                        if attempt<3:await asyncio.sleep(2)
                    if code:record.update(status='failed',failed_stage='resource_readback',error='See provider log for retained readback failures')
                    else:record['actual']=json.loads(observed_path.read_text());record['status']='deployed'
        record['end_utc_epoch']=time.time()
        (output/(provider+'.json')).write_text(json.dumps(record,indent=2,default=str)+'\n')
        return record
    results=await asyncio.gather(*(one(p) for p in (providers or config['providers'])))
    report={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_changes':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),'parallel_providers':True,'results':results}
    (output/'deployment.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=CONFIG);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deploy',action='store_true');p.add_argument('--provider',action='append');p.add_argument('--deploy-only',action='store_true')
    a=p.parse_args();load_dotenv(ROOT/'.env');print(json.dumps(asyncio.run(execute(yaml.safe_load(a.config.read_text()),a.output,deploy=a.deploy or a.deploy_only,providers=a.provider,deploy_only=a.deploy_only)),indent=2))
