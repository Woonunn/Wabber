import asyncio
from datetime import datetime,timezone
import json
import os
import random
import time
from openai import AsyncOpenAI,APIStatusError


class DeepSeekClient:
    def __init__(self,recorder,config):
        key=os.getenv('DEEPSEEK_API_KEY')
        if not key:raise ValueError('Configure DEEPSEEK_API_KEY outside Git')
        self.client=AsyncOpenAI(api_key=key,base_url='https://api.deepseek.com',max_retries=0,timeout=180)
        self.recorder=recorder;self.config=config;self.random=random.Random(config['seed']);self.tool_definitions=None

    async def decide(self,phase,iteration,messages):
        request={'model':self.config.get('name','deepseek-flash'),'messages':messages,'response_format':{'type':'json_object'},
                 **{k:self.config[k] for k in ('temperature','top_p','max_tokens')},
                 'extra_body':{'thinking':{'type':self.config.get('thinking','disabled')}}}
        if self.tool_definitions is not None:
            request.pop('response_format')
            definitions=json.loads(json.dumps(self.tool_definitions))
            for definition in definitions:
                definition['parameters'].get('properties',{}).pop('uid',None)
            request['tools']=[{'type':'function','function':d} for d in definitions]+[
                {'type':'function','function':{'name':'submit_result','description':'Finish the task by submitting the final answer object in the format required by the system prompt.',
                 'parameters':{'type':'object','properties':{'answer':{'type':'object'}},'required':['answer']}}}]
            request['parallel_tool_calls']=False
            if self.config.get('thinking','disabled')=='disabled':
                request['tool_choice']='required'
        self.recorder.emit('llm_start',phase=phase,iteration=iteration,request=request)
        start=time.perf_counter();started=datetime.now(timezone.utc).isoformat()
        for attempt in range(self.config.get('retries',2)+1):
            attempt_start=time.perf_counter();raw=None
            try:
                response=await self.client.chat.completions.create(**request)
                raw=response.model_dump(mode='json');content=response.choices[0].message.content
                calls=response.choices[0].message.tool_calls if self.tool_definitions is not None else None
                if calls:
                    if response.choices[0].finish_reason!='tool_calls':
                        raise ValueError('Expected one native tool call')
                    parsed=[{'tool':call.function.name,'args':json.loads(call.function.arguments)} for call in calls]
                    if any(c['tool']=='submit_result' for c in parsed):
                        if len(parsed)!=1:raise ValueError('Final answer cannot accompany pending tools')
                        decision=parsed[0]['args']
                    else:
                        decision=parsed[0] if len(parsed)==1 else {'calls':parsed}
                else:
                    if response.choices[0].finish_reason!='stop':raise ValueError('Incomplete model response')
                    decision=json.loads(content)
                if not isinstance(decision,dict):raise ValueError('Model decision must be a JSON object')
                self.recorder.emit('llm',phase=phase,iteration=iteration,request=request,response=raw,decision=decision,
                    requested_model=request['model'],actual_model=response.model,response_id=response.id,
                    usage=raw.get('usage'),input_tokens=(raw.get('usage') or {}).get('prompt_tokens'),
                    output_tokens=(raw.get('usage') or {}).get('completion_tokens'),started_utc=started,ended_utc=datetime.now(timezone.utc).isoformat(),
                    duration_s=time.perf_counter()-start,attempts=attempt+1,source='live_official_api')
                return decision
            except Exception as exc:
                status=exc.status_code if isinstance(exc,APIStatusError) else None
                retry=attempt<self.config.get('retries',2) and (status in (429,500,502,503,504) or status is None)
                delay=(2**attempt)+self.random.random()*.1 if retry else None
                self.recorder.emit('llm_error',phase=phase,iteration=iteration,attempt=attempt+1,status_code=status,
                    error_type=type(exc).__name__,error=str(exc).replace(self.client.api_key,'[redacted]')[:2000],duration_s=time.perf_counter()-attempt_start,
                    retry_delay_s=delay,request=request,response=raw,usage=raw.get('usage') if raw else None)
                if not retry:raise
                await asyncio.sleep(delay)
