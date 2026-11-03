import asyncio,json,re,time,uuid
from pathlib import Path
from atsuite.runtime import InvocationRequest,create_runtime_adapter
from atsuite.tracing.storage import read_text


class DivergenceError(RuntimeError):pass


def bcb_error(value):
    """Keep every failed test and assertion; omit unittest timing and incidental stack frames."""
    if not isinstance(value,str) or not value.startswith('Independent BCB tests failed:'):
        return value
    lines=[];stack=False
    for line in value.splitlines():
        if line.startswith('Traceback (most recent call last):'):
            stack=True;continue
        if stack:
            if not re.match(r'^[A-Za-z_]\w*(?:Error|Exception):',line):continue
            stack=False
        if re.match(r'^Ran \d+ tests? in [\d.]+s$',line):continue
        lines.append(line)
    return '\n'.join(lines)


def semantic(result):
    """Only control-significant output; timing and provider IDs are measured anew."""
    if not isinstance(result,dict):raise ValueError('Tool must return structured JSON')
    fields={k:result[k] for k in ('status','error_type','error','snapshot','checks','state_signature','state_names','branch_ref','step_id','inputs') if k in result}
    if 'error' in fields:fields['error']=bcb_error(fields['error'])
    if (fields.get('error_type')=='LookupError' and 'Attempted to load' in fields.get('error','')
            and 'https://www.nltk.org/data.html' in fields['error']):
        fields['error']=fields['error'].split('\n  Searched in:',1)[0].rstrip()
    if 'artifacts' in result:
        fields['artifacts']=[{k:a[k] for k in ('name','signature','comparison')} for a in sorted(result['artifacts'],key=lambda x:x['name'])]
    return fields


class ToolClient:
    def __init__(self,config,recorder,capture=None,calibration=False):
        if calibration and not capture:raise ValueError("Calibration requires a frozen Capture")
        self.calibration=calibration
        self.recorder=recorder;self.target_id=config['target_id'];self.expected_impl=config.get('expected_implementation_sha256');self.uid=uuid.uuid4().hex
        self.adapter=create_runtime_adapter(config['provider'],config['family'])
        self.adapter.connect({'targets':{self.target_id:{'endpoint':config['endpoint'],'family':config['family']}}})
        self.session=self.adapter.open_session(self.adapter.targets[self.target_id],self.uid)
        self.recorder.emit('tool_session_open',uid=self.uid,target_id=self.target_id,provider_session_id=self.session.provider_session_id,metadata=self.session.metadata)
        self.expected={}
        if capture:
            for line in read_text(Path(capture)/'events.jsonl').splitlines():
                event=json.loads(line)
                if event['kind']=='tool':
                    if event['key'] in self.expected:raise ValueError('Duplicate captured Tool key')
                    self.expected[event['key']]=event
        self.renderings=config.get('approved_renderings',{})
        for name,policy in self.renderings.items():
            signatures=policy['accepted_signatures']
            if not name.endswith('.png') or policy['reference_signature'] not in signatures or not all(len(s)==64 and all(c in '0123456789abcdef' for c in s) for s in signatures):raise ValueError('Invalid calibrated rendering policy')
            original={a['signature'] for e in self.expected.values() for a in e.get('result',{}).get('artifacts',[]) if a['name']==name}
            if original!={policy['reference_signature']}:raise ValueError('Rendering policy does not match the original Capture')
        self.replay=capture is not None;self.consumed=set();self.timeout=config.get('timeout',240);self.rpc_tasks=[]

    def comparable(self,value):
        if isinstance(value,list):return [self.comparable(v) for v in value]
        if not isinstance(value,dict):return value
        if 'error_type' in value and 'error' in value:
            value={**value,'error':semantic(value)['error']}
        result={k:self.comparable(v) for k,v in value.items()}
        policy=self.renderings.get(result.get('name'))
        if policy and result.get('comparison')=='decoded-rgba' and result.get('signature') in policy['accepted_signatures']:
            result['signature']=policy['reference_signature']
        return result

    def calibration_view(self,value):
        """Keep control flow and artifact identities while collecting new numeric evidence."""
        if isinstance(value,list):return [self.calibration_view(v) for v in value]
        if not isinstance(value,dict):return value
        # Explicit reference calibration can upgrade BCB pickle comparison; strict Replay cannot.
        method={'bcb_result.pkl':'bcb-return-v1','bcb_execution.json':'bcb-receipt-v1'}.get(value.get('name'))
        if method and 'comparison' in value:value={**value,'comparison':method}
        return {k:self.calibration_view(v) for k,v in value.items()
                if k not in {'state_signature','checks'} and not (k=='signature' and 'comparison' in value)}

    async def call(self,name,args,key,wave=None):
        if self.replay:
            expected=self.expected.get(key)
            compare=self.calibration_view if self.calibration and name=='merge_artifacts' else self.comparable
            if expected is None or expected['tool_name']!=name or compare(expected['args'])!=compare(args):
                raise DivergenceError('Replay Tool parameters diverged: '+key)
        start=time.perf_counter()
        request=InvocationRequest(target_id=self.target_id,tool_name=name,args=args,uid=self.uid,
            call_id=uuid.uuid4().hex,session=self.session,timeout=self.timeout,run_id=self.recorder.request_id)
        self.recorder.emit('tool_start',key=key,tool_name=name,args=args,wave=wave,call_id=request.call_id,
            runtime_uid=self.uid,target_id=self.target_id,provider_session_id=self.session.provider_session_id)
        async def invoke():
            try:
                response=await asyncio.to_thread(self.adapter.invoke,request)
            except BaseException as exc:
                self.recorder.emit('tool_error',key=key,tool_name=name,error_type=type(exc).__name__,error=str(exc))
                raise
            self.recorder.emit('tool',key=key,tool_name=name,args=args,wave=wave,result=response.result,
                transport=response.to_dict(),duration_s=time.perf_counter()-start)
            return response
        task=asyncio.create_task(invoke());self.rpc_tasks.append(task)
        response=await asyncio.shield(task)
        result=response.result
        if name=='prepare_workspace' and self.expected_impl and isinstance(result,dict) and result.get('status')=='ok':
            if result.get('execution_environment',{}).get('implementation_sha256')!=self.expected_impl:
                raise DivergenceError('Remote Tool code does not match the audited deployment image')
        structured_failure=isinstance(result,dict) and result.get('status') in {'error','timeout'} and result.get('error_type')
        if response.status!='ok' and not structured_failure:raise RuntimeError(f'Tool transport {response.status}: {response.error}')
        if self.replay:
            compare=self.calibration_view if self.calibration else self.comparable
            if compare(semantic(result))!=compare(semantic(expected['result'])):
                self.recorder.emit('divergence',key=key,expected=semantic(expected['result']),actual=semantic(result))
                raise DivergenceError('Actual Tool output incompatible with Capture: '+key)
            if semantic(result)!=semantic(expected['result']):
                if self.calibration:self.recorder.emit('environment_reference_difference',key=key,expected=semantic(expected['result']),actual=semantic(result))
                else:self.recorder.emit('calibrated_rendering_equivalence',key=key,policy=self.renderings)
            self.consumed.add(key)
        return result

    def close(self):
        self.adapter.close_session(self.session)
        if hasattr(self.adapter,'close'):self.adapter.close()

    async def aclose(self):
        if self.rpc_tasks:await asyncio.gather(*self.rpc_tasks,return_exceptions=True)
        self.close()

    def assert_consumed(self):
        if self.replay and self.consumed!=set(self.expected):raise DivergenceError('Replay did not execute every captured Tool call')
