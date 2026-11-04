"""Canonical state follows ATSuite MCP/FaaS semantics; branches never commit it."""
import hashlib,io,json,os,pickle,shutil,subprocess,tarfile,tempfile,time,uuid,threading,platform
from pathlib import Path
from contextvars import ContextVar
from functools import wraps
_settings=Path(__file__).with_name('cloud_settings.json')
if _settings.exists():
    for _key,_value in json.loads(_settings.read_text()).items():
        os.environ.setdefault(_key,str((Path(__file__).parent/_value).resolve()) if _key in {'WABBER_DATA_ROOT','WABBER_EVALUATION_MAPPING','WABBER_APPROVED_TRACE'} else str(_value))
from atsuite_sdk.abstract import registry
from atsuite_sdk.state import get_state_runtime,register_state_object
try:
    from .logical_state import StateObservation,json_size
    from .guards import identifier,relative_file,safe_files
except ImportError:
    from logical_state import StateObservation,json_size
    from guards import identifier,relative_file,safe_files

try:
    from .frozen_policy import check_preparation,check_analysis
    from .native_kernel import NativeKernel
    from .resident_kernel import ResidentKernel
except ImportError:
    from frozen_policy import check_preparation,check_analysis
    from native_kernel import NativeKernel
    from resident_kernel import ResidentKernel


class AnalysisFailure(RuntimeError):
    def __init__(self,result):
        super().__init__(result.get('error','Tool execution failed'))
        self.result=result


class Workspace:
    def __init__(self):self.version=0;self.task_id='';self.inputs=[];self.artifacts=[]


_mcp_kernel=None
_faas_kernels={}
_faas_kernel_users={}
_faas_kernel_lock=threading.Lock()
_environment_id=uuid.uuid4().hex
_first_task=True
_environment_lock=threading.Lock()


workspace=register_state_object('agent_workspace',Workspace(),session_scoped=True)


_observations=ContextVar('datasci_logical_state')


def workspace_metadata():
    return {key:getattr(workspace,key) for key in ('version','task_id','inputs','artifacts')}


def branch_metadata(result):
    # Runtime measurements and stdout are telemetry, not reusable branch state.
    return {key:result[key] for key in ('artifacts','state_signature','state_names') if key in result}


def observe_workspace(access):
    """Observe the same metadata boundary inside either Runtime's execution lock."""
    def decorate(function):
        @wraps(function)
        def measured(*args,**kwargs):
            observation=StateObservation();token=_observations.set(observation)
            try:
                observation.record('workspace/metadata','read',lambda:json_size(workspace_metadata()))
                result=function(*args,**kwargs)
                if access=='rw':
                    observation.record('workspace/metadata','write',lambda:json_size(workspace_metadata()))
                return observation.attach(result)
            except BaseException as exc:
                if not (access=='r' and isinstance(exc,AnalysisFailure) and exc.result.get('logical_state',{}).get('complete') is True):
                    observation.complete=False
                    observation.errors.append('Tool failed: '+type(exc).__name__)
                if isinstance(getattr(exc,'result',None),dict):observation.attach(exc.result)
                raise
            finally:_observations.reset(token)
        return measured
    return decorate


def observe_artifacts(root,operation):
    # Full visible artifact snapshot is the externalization unit, not individual open() calls.
    for path in sorted(root.rglob('*')):
        if path.is_file():
            _observations.get().record('canonical/artifacts/'+path.relative_to(root).as_posix(),operation,lambda:path.stat().st_size)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def store(key,source=None,destination=None):
    counter='branch_write_bytes' if source is not None else 'branch_read_bytes'
    try:
        provider=os.getenv('ATSUITE_BRANCH_PROVIDER') or os.getenv('PROVIDER','')
        if get_state_runtime().runtime=='mcp':provider=''
        if provider:
            from atsuite_sdk.storage import create_storage
            storage=create_storage(provider,bucket=os.environ['PROJECT'])
            if source is not None:
                if not storage.try_create(key,Path(source).read_bytes()):raise ValueError('Immutable branch already exists')
            else:storage.download(key,str(destination))
        else:
            target=get_state_runtime().root/'branch_artifacts'/key
            if source is not None:
                target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix('.tmp-'+uuid.uuid4().hex)
                shutil.copyfile(source,temp)
                try:os.link(temp,target)
                finally:temp.unlink(missing_ok=True)
            else:shutil.copyfile(target,destination)
    except BaseException:
        get_state_runtime().record_transfer_bytes(counter,None)
        raise
    get_state_runtime().record_transfer_bytes(counter,
        Path(source if source is not None else destination).stat().st_size)



def unpack(source,dest):
    with tarfile.open(source) as archive:
        members=archive.getmembers()
        if sum(m.size for m in members)>1024**3 or len(members)>256:raise ValueError('Artifact bundle quota exceeded')
        for member in members:
            name=relative_file(member.name)
            if not member.isfile():raise ValueError('Artifact bundles contain regular files only')
            p=dest/name;p.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as incoming,p.open('wb') as outgoing:shutil.copyfileobj(incoming,outgoing)


def sandbox(request,parent,inputs=None,branches=None):
    if get_state_runtime().runtime=='mcp':
        if _mcp_kernel is None:raise RuntimeError('Prepare the MCP Workspace first')
        return _mcp_kernel.run(request,parent,inputs,branches)
    if os.getenv('WABBER_EXECUTION_BACKEND')=='native':
        inputs=inputs or Path(os.environ['WABBER_DATA_ROOT'])/workspace.task_id
        key=(str(get_state_runtime().root.resolve()),str(Path(inputs).resolve()))
        with _faas_kernel_lock:
            # Keep one idle stateless interpreter; 100 task-specific heaps must not accumulate in a warm FaaS instance.
            for old in list(_faas_kernels):
                if old!=key and not _faas_kernel_users.get(old,0):
                    _faas_kernels.pop(old).close();_faas_kernel_users.pop(old,None)
            kernel=_faas_kernels.get(key)
            if kernel is None or kernel.closed:
                kernel=NativeKernel(get_state_runtime().root,inputs,resident=False)
                _faas_kernels[key]=kernel
            _faas_kernel_users[key]=_faas_kernel_users.get(key,0)+1
        try:return kernel.run(request,parent,inputs,branches)
        finally:
            with _faas_kernel_lock:
                _faas_kernel_users[key]-=1
                if not _faas_kernel_users[key] and len(_faas_kernels)>1:
                    _faas_kernels.pop(key).close();_faas_kernel_users.pop(key,None)
    image=os.environ['WABBER_SANDBOX_IMAGE']
    if not image.startswith('sha256:'):raise ValueError('Pin the sandbox image ID')
    with tempfile.TemporaryDirectory(prefix='wabber-sandbox-') as directory:
        root=Path(directory);(root/'out').mkdir();(root/'request.json').write_text(json.dumps(request));container_name='wabber-agent-'+uuid.uuid4().hex
        command=['docker','run','--name',container_name,'--network','none','--read-only','--cpus','2','--memory','4g','--memory-swap','4g',
            '--pids-limit','128','--cap-drop','ALL','--security-opt','no-new-privileges','--user',f'{os.getuid()}:{os.getgid()}',
            '--ulimit','nofile=256:256','--ulimit','fsize=536870912:536870912','--log-driver','none',
            '--tmpfs','/tmp:rw,nosuid,size=256m','-v',f'{root}:/work:rw','-v',f'{parent}:/snapshot:ro']
        if inputs:
            command+=['-v',f'{inputs}:/inputs:ro','-v',f'{inputs}:/work/out/inputs_dir:ro']
            for source in inputs.rglob('*'):
                if source.is_file():
                    name=source.relative_to(inputs).as_posix()
                    if name not in request.get('outputs',[]):command+=['-v',f'{source}:/work/out/{name}:ro']
        for source in (parent/'artifacts').rglob('*'):
            if source.is_file():
                name=source.relative_to(parent/'artifacts').as_posix()
                if name not in request.get('outputs',[]):command+=['-v',f'{source}:/work/out/{name}:ro']
        command+=['-v',f'{root / "out"}:/work/out/output_dir:rw']
        if branches:command+=['-v',f'{branches}:/branches:ro']
        command+=[image]
        started=time.perf_counter();stderr_path=root/'docker-errors.txt'
        with stderr_path.open('w') as errors:process=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=errors)
        try:
            while process.poll() is None:
                if time.perf_counter()-started>float(os.getenv('WABBER_CODE_TIMEOUT','180')):
                    raise TimeoutError('Generated code exceeded sandbox time limit')
                safe_files(root)
                # Resource monitoring only, not benchmark CPU work.
                time.sleep(.05)
            safe_files(root)
            result_path=root/'result.json'
            if not result_path.exists():raise RuntimeError(f'Sandbox exited {process.returncode} without a result: '+stderr_path.read_text(errors='replace')[:16000])
            if result_path.stat().st_size>1024**2:raise ValueError('Oversized sandbox response')
            result=json.loads(result_path.read_text());result['sandbox_image']=image;result['container_name']=container_name
            result['sandbox_wall_s']=time.perf_counter()-started
            if result['status']!='ok':return result,None
            # Host never deserializes Python state or models.
            bundle=root/'bundle.tar'
            with tarfile.open(bundle,'w') as archive:
                allowed={'state.pkl','stdout.txt',*request.get('outputs',[])}
                for p in safe_files(root/'out'):
                    relative=p.relative_to(root/'out').as_posix()
                    if relative in allowed:archive.add(p,arcname=relative,recursive=False)
            fd,name=tempfile.mkstemp(prefix='wabber-bundle-',suffix='.tar');os.close(fd);saved=Path(name);shutil.copyfile(bundle,saved)
            return result,saved
        finally:
            subprocess.run(['docker','rm','-f',container_name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            process.wait(timeout=10)


def current(uid,version):
    if version!=workspace.version:raise ValueError('Stale Workspace generation')
    return get_state_runtime().session_dir(uid)/f'v{version}'


@registry.tool(access='rw',domain='agent_workspace')
@observe_workspace('rw')
def prepare_workspace(task_id:str,dataset_refs:list,uid:str)->dict:
    global _mcp_kernel,_first_task
    identifier(task_id)
    if os.getenv('WABBER_NATIVE_CONFINEMENT')=='frozen_trace':check_preparation(task_id,dataset_refs)
    if workspace.task_id:raise ValueError('Workspace already prepared; create a new Session')
    inputs=Path(os.environ['WABBER_DATA_ROOT'])/task_id
    for ref in dataset_refs:
        name=relative_file(ref['name']);path=inputs/name
        if not path.resolve().is_relative_to(inputs.resolve()) or sha(path)!=ref['sha256']:raise ValueError('Dataset checksum mismatch')
    root=get_state_runtime().session_dir(uid)/'v0';(root/'artifacts').mkdir(parents=True,exist_ok=False)
    if get_state_runtime().runtime!='mcp':(root/'state.pkl').write_bytes(pickle.dumps({},protocol=5))
    _observations.get().record('canonical/state','write',lambda:len(pickle.dumps({},protocol=5)))
    workspace.task_id=task_id;workspace.inputs=dataset_refs
    with _environment_lock:
        first_task=_first_task;_first_task=False
    result={'status':'ok','snapshot':0,'workspace_id':uid,'inputs':dataset_refs,
        'execution_environment':{'environment_id':_environment_id,'first_task_in_process':first_task,
          'implementation_sha256':sha(Path(__file__)),'python':platform.python_version(),'architecture':platform.machine(),
          'confinement':os.getenv('WABBER_NATIVE_CONFINEMENT','docker')}}
    if get_state_runtime().runtime=='mcp':
        if os.getenv('WABBER_EXECUTION_BACKEND')=='native':
            _mcp_kernel=NativeKernel(root.parent,inputs,resident=True)
        else:_mcp_kernel=ResidentKernel(root.parent,inputs,os.environ['WABBER_SANDBOX_IMAGE'])
        result['kernel']={'container_name':_mcp_kernel.name,'pid':_mcp_kernel.pid,'session_start_s':_mcp_kernel.start_s,'lifecycle':'resident'}
    return result


@registry.tool(access='r',domain='agent_workspace')
@observe_workspace('r')
def execute_analysis(step_id:str,code:str,outputs:list,snapshot:int,seed:int,call_id:str,uid:str)->dict:
    identifier(step_id);identifier(call_id)
    if os.getenv('WABBER_NATIVE_CONFINEMENT')=='frozen_trace':check_analysis(workspace.task_id,step_id,code,outputs,snapshot,seed)
    for name in outputs:relative_file(name)
    parent=current(uid,snapshot);inputs=Path(os.environ['WABBER_DATA_ROOT'])/workspace.task_id
    observe_artifacts(parent/'artifacts','read')
    result,bundle=sandbox({'mode':'analyze','task_id':workspace.task_id,'code':code,'outputs':outputs,'seed':seed},parent,inputs=inputs)
    _observations.get().include(result)
    for event in _observations.get().accesses:
        if event['object_id']=='branch/state':event['object_id']='branch/'+step_id+'/state'
    if result['status']!='ok':raise AnalysisFailure(result)
    if bundle:
        try:
            for artifact in result['artifacts']:
                _observations.get().record('branch/'+step_id+'/artifacts/'+artifact['name'],'write',lambda:artifact['bytes'])
            _observations.get().record('branch/'+step_id+'/metadata','write',lambda:json_size(branch_metadata(result)))
            with tarfile.open(bundle,'a') as archive:
                raw=json.dumps(result).encode();info=tarfile.TarInfo('branch_manifest.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
            store(f'{identifier(uid)}/{call_id}.tar',source=bundle)
        finally:bundle.unlink()
        result.update(branch_ref=call_id,step_id=step_id,snapshot=snapshot)
    return result


@registry.tool(access='rw',domain='agent_workspace')
@observe_workspace('rw')
def merge_artifacts(branches:list,snapshot:int,uid:str)->dict:
    if not branches:raise ValueError('Cannot commit an empty branch batch')
    parent=current(uid,snapshot);root=get_state_runtime().session_dir(uid)
    observe_artifacts(parent/'artifacts','read')
    with tempfile.TemporaryDirectory(prefix='wabber-merge-') as directory:
        branch_root=Path(directory);names=[];new_artifacts=[];seen=set()
        for ref in branches:
            name=identifier(ref['step_id']);call=identifier(ref['branch_ref']);names.append(name)
            dest=branch_root/name;dest.mkdir();archive=branch_root/(name+'.tar');store(f'{identifier(uid)}/{call}.tar',destination=archive);unpack(archive,dest)
            metadata=json.loads((dest/'branch_manifest.json').read_text())
            _observations.get().record('branch/'+name+'/metadata','read',lambda:json_size(branch_metadata(metadata)))
            expected=[{k:a[k] for k in ('name','signature','comparison')} for a in metadata['artifacts']]
            if expected!=ref['artifacts']:raise ValueError('Branch semantic outputs changed before merge')
            for a in metadata['artifacts']:
                output=relative_file(a['name'])
                if output in seen:raise ValueError('Conflicting branch output writes')
                seen.add(output)
                if sha(dest/output)!=a['sha256']:raise ValueError('Branch artifact corrupted before merge')
                _observations.get().record('branch/'+name+'/artifacts/'+output,'read',lambda:(dest/output).stat().st_size)
                new_artifacts.append(a)
        result,bundle=sandbox({'mode':'merge','steps':names},parent,branches=branch_root)
        _observations.get().include(result)
        if result['status']!='ok':raise AnalysisFailure(result)
        target=root/f'v{snapshot+1}';staging=root/('.pending-'+uuid.uuid4().hex)
        staging.mkdir();(staging/'artifacts').mkdir();resident_committed=False
        try:
            unpack(bundle,staging)
            for p in (parent/'artifacts').rglob('*'):
                if p.is_file():
                    dest=staging/'artifacts'/p.relative_to(parent/'artifacts');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
            for ref in branches:
                for a in ref['artifacts']:
                    dest=staging/'artifacts'/a['name'];dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(branch_root/ref['step_id']/a['name'],dest)
            if target.exists():shutil.rmtree(target)
            staging.replace(target)
            if get_state_runtime().runtime=='mcp':
                committed,_=_mcp_kernel.run({'mode':'commit','token':result['merge_token']},target)
                if committed['status']!='ok':raise AnalysisFailure(committed)
                resident_committed=True
            workspace.artifacts=[a for a in workspace.artifacts if a['name'] not in seen]+new_artifacts
            workspace.version=snapshot+1
            observe_artifacts(target/'artifacts','write')
            shutil.rmtree(parent)
            return {'status':'ok','snapshot':workspace.version,'artifacts':workspace.artifacts,'state_signature':result['state_signature'],
                'kernel_execution':{k:result[k] for k in ('state_lifecycle','worker_usage','sandbox_wall_s','container_name','resident_kernel_pid','exec_started_monotonic','exec_ended_monotonic') if k in result}}
        finally:
            if get_state_runtime().runtime=='mcp' and not resident_committed:
                _mcp_kernel.run({'mode':'abort','token':result['merge_token']},parent)
            bundle.unlink()
            if staging.exists():shutil.rmtree(staging)


@registry.tool(access='r',domain='agent_workspace')
@observe_workspace('r')
def read_workspace(snapshot:int,uid:str)->dict:
    root=current(uid,snapshot)
    return {'status':'ok','snapshot':snapshot,'artifacts':workspace.artifacts,
        'state_bytes':(root/'state.pkl').stat().st_size if (root/'state.pkl').exists() else None,
        'state_lifecycle':'resident' if get_state_runtime().runtime=='mcp' else 'restore-persist'}


@registry.tool(access='r',domain='agent_workspace')
@observe_workspace('r')
def validate_result(outputs:list,snapshot:int,uid:str)->dict:
    for name in outputs:relative_file(name)
    observe_artifacts(current(uid,snapshot)/'artifacts','read')
    result,bundle=sandbox({'mode':'validate','outputs':outputs,'task_id':workspace.task_id,
        'artifact_evidence':{a['name']:a.get('evidence') for a in workspace.artifacts}},current(uid,snapshot),
        inputs=Path(os.environ['WABBER_DATA_ROOT'])/workspace.task_id)
    _observations.get().include(result)
    if bundle:bundle.unlink()
    if result['status']!='ok':raise AnalysisFailure(result)
    if workspace.task_id.startswith('bcb'):
        result['evaluation_basis']={'mode':'upstream_bcb_tests','official_pass':None,
            'test_sha256':result['checks']['test_sha256'],
            'scope':'Original invocation plus full upstream tests; not official TMC/CR scoring'}
        return result
    mapping_path=Path(os.environ['WABBER_EVALUATION_MAPPING']) if 'WABBER_EVALUATION_MAPPING' in os.environ else Path(__file__).resolve().parents[2]/'tasks/evaluation_alignment.json'
    mapping=json.loads(mapping_path.read_text())
    task=next(t for t in mapping['tasks'] if t['task_id']==workspace.task_id)
    result['evaluation_basis']={'mode':'custom_prompt_checks','official_pass':None,
        'source_commit_sha':mapping['source_commit_sha'],'mapping_sha256':sha(mapping_path),
        'metrics':[{'metric':m['metric'],'coverage':m['coverage']} for m in task['metrics']]}
    return result
