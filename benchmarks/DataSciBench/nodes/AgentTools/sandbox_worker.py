"""Runs only inside a credential-free Docker sandbox, including unpickling state."""
import hashlib,json,os,pickle,sys,time,traceback,resource
from pathlib import Path
from types import ModuleType
from joblib.externals import cloudpickle
from kernel import execute_code
from logical_state import StateObservation


def canonical(value,depth=0):
    import numpy as np
    import pandas as pd
    if isinstance(value,type):return {'python_type':value.__module__+'.'+value.__qualname__}
    if type(value).__module__.startswith('torch'):
        import torch
        if isinstance(value,torch.Tensor):return canonical(value.detach().cpu().numpy(),depth+1)
        if isinstance(value,torch.nn.Module):return {'torch_module':type(value).__name__,'weights':canonical(dict(value.state_dict()),depth+1)}
    if depth>20:raise ValueError('State cannot be compared deterministically')
    if isinstance(value,np.bool_):return bool(value)
    if value is None or isinstance(value,(str,bool,int)):return value
    if isinstance(value,(float,np.floating)):
        return format(float(value),'.8g')
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,(np.ndarray,pd.DataFrame,pd.Series)):
        if isinstance(value,pd.DataFrame):
            return {'frame':hashlib.sha256(value.to_json(orient='split',double_precision=8,date_format='iso').encode()).hexdigest()}
        if isinstance(value,pd.Series):return {'series':value.to_json(double_precision=8,date_format='iso')}
        if value.dtype.fields:return {'dtype':str(value.dtype),'fields':{k:canonical(value[k],depth+1) for k in value.dtype.names}}
        if np.issubdtype(value.dtype,np.number):
            return {'shape':list(value.shape),'dtype':str(value.dtype),'sha256':hashlib.sha256(np.round(value.astype(float),8).tobytes()).hexdigest()}
        return {'shape':list(value.shape),'values':value.tolist()}
    if isinstance(value,dict):return {str(k):canonical(v,depth+1) for k,v in sorted(value.items(),key=lambda x:str(x[0]))}
    if isinstance(value,(list,tuple)):return [canonical(v,depth+1) for v in value]
    if isinstance(value,(set,frozenset)):return sorted(canonical(v,depth+1) for v in value)
    if hasattr(value,'get_params'):
        return {'class':type(value).__module__+'.'+type(value).__name__,
                'parameters':canonical(value.get_params(deep=False),depth+1),
                'fitted':canonical({k:v for k,v in vars(value).items() if k.endswith('_')},depth+1)}
    if hasattr(value,'__getstate__'):
        state=value.__getstate__()
        if state is not None:return {'class':type(value).__name__,'state':canonical(state,depth+1)}
    raise ValueError('Unsupported state comparison type: '+type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def artifact(path,out,task_id=None,evidence=None):
    import pandas as pd
    suffix=path.suffix.lower();raw=path.read_bytes()
    signature=hashlib.sha256(raw).hexdigest();method='sha256'
    if suffix=='.csv':signature=digest(canonical(pd.read_csv(path)));method='csv-round-8-decimal-places'
    elif path.name=='bcb_result.pkl':
        from bcb_checks import return_signature,timing_signature
        if task_id=='bcb823':
            if not evidence:raise ValueError('Missing timing invocation evidence')
            signature=timing_signature(cloudpickle.loads(raw),evidence);method='bcb-return-timing-v2'
        else:
            clock_window=(evidence or {}).get('invocation_clock_window')
            if task_id=='bcb61' and clock_window is None:raise ValueError('Missing invocation clock evidence')
            signature=return_signature(cloudpickle.loads(raw),clock_window=clock_window if task_id=='bcb61' else None)
            method='bcb-return-clock-v2' if task_id=='bcb61' else 'bcb-return-v1'
    elif path.name=='bcb_execution.json':
        receipt=json.loads(raw);receipt.pop('return_sha256')
        signature=digest(receipt);method='bcb-receipt-v1'
    elif suffix=='.json':signature=digest(json.loads(raw));method='json'
    elif suffix=='.png':
        from PIL import Image
        with Image.open(path) as im:
            im=im.convert('RGBA');signature=digest({'shape':im.size,'pixels':hashlib.sha256(im.tobytes()).hexdigest()})
        method='decoded-rgba'
    elif suffix in ('.pt','.pth'):
        import torch
        signature=digest(canonical(torch.load(path,map_location='cpu',weights_only=True)));method='torch-weights-only-canonical'
    elif suffix=='.pdf':
        from pypdf import PdfReader
        reader=PdfReader(path);signature=digest([(p.extract_text(),[hashlib.sha256(i.data).hexdigest() for i in p.images]) for p in reader.pages]);method='pdf-text-and-images'
    result={'name':path.relative_to(out).as_posix(),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
            'signature':signature,'comparison':method}
    if method in {'bcb-return-clock-v2','bcb-return-timing-v2'}:result['evidence']=evidence
    return result


def run(request,parent,out,inputs=Path('/inputs'),branches=Path('/branches'),shared=None,resident=False):
    out.mkdir(exist_ok=True)
    started=time.perf_counter();usage=resource.getrusage(resource.RUSAGE_SELF);cpu_start=usage.ru_utime+usage.ru_stime;result={'status':'ok'}
    observation=StateObservation();state_ready=False
    try:
        state_path=parent/'state.pkl'
        restored=shared is None
        restore_start=time.perf_counter()
        if restored:shared=cloudpickle.loads(state_path.read_bytes()) if state_path.exists() else {}
        result['state_lifecycle']={'mode':'resident' if resident else 'restore-persist',
            'canonical_restore_performed':restored,'canonical_restore_s':time.perf_counter()-restore_start if restored else None,
            'canonical_restore_bytes':state_path.stat().st_size if restored and state_path.exists() else None,
            'canonical_save_performed':False,'canonical_save_s':None,'canonical_save_bytes':None}
        observation.record('canonical/state', 'read', lambda: len(cloudpickle.dumps(shared,protocol=5)))
        state_ready=True
        if request['mode']=='merge':
            for step in request['steps']:
                observation.record('branch/'+step+'/state', 'read', lambda: (branches/step/'state.pkl').stat().st_size)
                shared[step]=cloudpickle.loads((branches/step/'state.pkl').read_bytes())
            observation.record('canonical/state', 'write', lambda: len(cloudpickle.dumps(shared,protocol=5)))
            if not resident:
                save_start=time.perf_counter()
                (out/'state.pkl').write_bytes(cloudpickle.dumps(shared,protocol=5))
                result['state_lifecycle'].update(canonical_save_performed=True,canonical_save_s=time.perf_counter()-save_start,canonical_save_bytes=(out/'state.pkl').stat().st_size)
            result.update(state_signature=digest(canonical(shared)))
        elif request['mode']=='analyze':
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib.figure import Figure
            recorded={};original_save=Figure.savefig
            def save(figure,filename,*args,**kwargs):
                result=original_save(figure,filename,*args,**kwargs)
                path=Path(filename).resolve()
                if path.is_relative_to(out/'output_dir'):path=out/path.relative_to(out/'output_dir')
                name=path.relative_to(out).as_posix()
                bars=[];lines=[];scatter=[]
                for ax in figure.axes:
                    for patch in ax.patches:
                        if hasattr(patch,'get_height'):
                            bars.append({'height':float(patch.get_height()),'x':float(patch.get_x()),'width':float(patch.get_width()),'face':list(patch.get_facecolor()),'edge':list(patch.get_edgecolor())})
                    for line in ax.lines:
                        y=line.get_ydata()
                        lines.append({'label':line.get_label(),'style':line.get_linestyle(),'y':[float(x) if __import__('math').isfinite(float(x)) else None for x in y] if len(y)<=64 else [],'count':len(y)})
                    for collection in ax.collections:
                        offsets=collection.get_offsets()
                        if len(offsets)<=64 and hasattr(collection,'get_sizes'):
                            scatter.append({'y':[float(x) for x in offsets[:,1]],'sizes':collection.get_sizes().tolist(),'alpha':collection.get_alpha(),'face':collection.get_facecolors().tolist()})
                recorded[name]={'bars':bars,'lines':lines,'scatter':scatter}
                return result
            Figure.savefig=save
            def artifact_path(name):
                p=((parent/'artifacts')/name).resolve()
                if not p.is_relative_to((parent/'artifacts')):raise ValueError('Invalid artifact path')
                return str(p)
            scope={'__name__':'__main__','__builtins__':__builtins__,'inputs_dir':inputs,
                   'output_dir':out,'shared':shared,'state_out':{},'artifact_path':artifact_path,'seed':request['seed']}
            import random,numpy as np
            random.seed(request['seed']);np.random.seed(request['seed'])
            training={};hooks=[]
            if request.get('task_id')=='csv_excel_29':
                from extended_checks import observe_training
                training,hooks=observe_training(request['seed'])
            os.chdir(out)
            execution=execute_code(request['code'],scope,out/'stdout.txt','/work/analysis.py')
            for hook in hooks:hook.remove()
            if execution['status']!='ok':
                error_type=execution['error'].strip().splitlines()[-1].split(':',1)[0]
                result.update(status='error',error_type=error_type,error=execution['error'].strip().splitlines()[-1].split(': ',1)[-1],traceback=execution['error'],stdout=execution['output'][:16000])
                # The failed branch exports no persistent state; its input read was measured.
                return finish(observation.attach(result),started,out,cpu_start)
            invocation_evidence=None
            if request.get('task_id','').startswith('bcb') and 'solution.py' in request['outputs']:
                from bcb_checks import invoke
                source=(out/'solution.py').read_text()
                invocation_start=time.perf_counter()
                clock_start=time.time()
                timing_observations={} if request.get('task_id')=='bcb823' else None
                value=invoke(source,inputs,timing_observations=timing_observations)
                invocation_evidence=(timing_observations if timing_observations is not None
                    else {'invocation_clock_window':[clock_start,time.time()]})
                result['bcb_invocation_s']=time.perf_counter()-invocation_start
                (out/'bcb_result.pkl').write_bytes(cloudpickle.dumps(value,protocol=5))
                (out/'bcb_execution.json').write_text(json.dumps({
                    'solution_sha256':hashlib.sha256(source.encode()).hexdigest(),
                    'input_sha256':hashlib.sha256((inputs/'fixture_args.pkl').read_bytes()).hexdigest(),
                    'return_sha256':hashlib.sha256((out/'bcb_result.pkl').read_bytes()).hexdigest(),
                    'invocations':1},sort_keys=True))
                del value
            exported=scope['state_out']
            if not isinstance(exported,dict):raise ValueError('state_out must be a dict')
            export_start=time.perf_counter()
            try:
                exported_bytes=cloudpickle.dumps(exported,protocol=5)
            except Exception:
                observation.complete=False
                observation.errors.append('Branch snapshot serialization failed')
                raise
            (out/'state.pkl').write_bytes(exported_bytes)
            del exported_bytes
            observation.record('branch/state', 'write', lambda: (out/'state.pkl').stat().st_size)
            result['state_lifecycle'].update(branch_export_s=time.perf_counter()-export_start,branch_export_bytes=(out/'state.pkl').stat().st_size)
            files=[artifact(out/name,out,request.get('task_id'),invocation_evidence) for name in request['outputs']]
            for f in files:
                if f['name'] in recorded:f['evidence']=recorded[f['name']]
                if f['name']=='model_checkpoint.pt' and request.get('task_id')=='csv_excel_29':f['evidence']={'optimizer_steps':list(training.values())}
            result.update(artifacts=files,state_signature=digest(canonical(exported)),state_names=sorted(exported),
                          stdout=(out/'stdout.txt').read_text(errors='replace')[:16000])
        elif request['mode']=='validate':
            os.chdir(out)
            files=[]
            for name in request['outputs']:
                p=(parent/'artifacts')/name
                if not p.is_file() or p.stat().st_size==0:raise ValueError('Missing or empty required output: '+name)
                # File parsing stays inside the Tool sandbox.
                import shutil
                target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
                evidence=request.get('artifact_evidence',{}).get(name) or {}
                files.append(artifact(target,out,request.get('task_id'),evidence))
            from result_checks import validate
            checks=validate(request['task_id'],(parent/'artifacts'),inputs,request.get('artifact_evidence',{}))
            result.update(artifacts=files,state_signature=digest(canonical(shared)),checks=checks)
        else:raise ValueError('Unknown sandbox mode')
    except BaseException as exc:
        # Business failure does not invalidate the snapshot accesses already measured.
        if not state_ready:
            observation.complete=False
            observation.errors.append('State restore failed: '+type(exc).__name__)
        result.update(status='error',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
                stdout=(out/'stdout.txt').read_text(errors='replace')[:16000] if (out/'stdout.txt').exists() else '')
    return finish(observation.attach(result),started,out,cpu_start)


def finish(result,started,out,cpu_start):
    usage=resource.getrusage(resource.RUSAGE_SELF)
    result['worker_usage']={'cpu_seconds':usage.ru_utime+usage.ru_stime-cpu_start,
        'cpu_scope':'isolated_worker_process','cpu_complete':False,'cpu_interval':'call_delta',
        'includes_children':False,'includes_other_threads':True,'pid':os.getpid(),
        'peak_rss_bytes':usage.ru_maxrss*1024,'rss_scope':'worker process high-water mark; resident merge process includes prior calls'}
    result.update(exec_started_monotonic=started,exec_ended_monotonic=time.perf_counter(),pid=os.getpid())
    (out.parent/'result.json').write_text(json.dumps(result,allow_nan=False)+'\n')
    return result


def main():
    request=json.loads(Path('/work/request.json').read_text())
    result=run(request,Path('/snapshot'),Path('/work/out'))
    if result['status']!='ok':sys.exit(1)


if __name__=='__main__':main()
