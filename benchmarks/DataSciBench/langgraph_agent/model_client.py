import asyncio
from abc import ABC, abstractmethod
from datetime import datetime,timezone
import json
import os
import random
import time
from openai import AsyncOpenAI,APIStatusError


class ModelClient(ABC):
    def __init__(self,recorder,config,*,api_key_env,base_url):
        key=os.getenv(api_key_env)
        if not key:raise ValueError(f'Configure {api_key_env} outside Git')
        self.client=AsyncOpenAI(api_key=key,base_url=base_url,max_retries=0,timeout=180)
        self.recorder=recorder;self.config=config;self.random=random.Random(config['seed']);self.tool_definitions=None

    @abstractmethod
    def model_parameters(self):
        """Return only parameters supported by this model provider."""
        raise NotImplementedError

    async def aclose(self):
        await self.client.close()

    async def decide(self,phase,iteration,messages):
        request={'model':self.config.get('name',self.default_model),'messages':messages,
                 'response_format':{'type':'json_object'},**self.model_parameters()}
        if self.tool_definitions is not None:
            request.pop('response_format')
            definitions=json.loads(json.dumps(self.tool_definitions))
            for definition in definitions:
                definition['parameters'].get('properties',{}).pop('uid',None)
            request['tools']=[{'type':'function','function':d} for d in definitions]+[
                {'type':'function','function':{'name':'submit_result','description':'Finish the task by submitting the final answer object in the format required by the system prompt.',
                 'parameters':{'type':'object','properties':{'answer':{'type':'object'}},'required':['answer']}}}]
            request['parallel_tool_calls']=False
            if self.require_tool_call:
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


class DeepSeekClient(ModelClient):
    default_model='deepseek-flash'

    def __init__(self,recorder,config):
        super().__init__(recorder,config,api_key_env='DEEPSEEK_API_KEY',base_url='https://api.deepseek.com')
        self.require_tool_call=config.get('thinking','disabled')=='disabled'

    def model_parameters(self):
        return {**{k:self.config[k] for k in ('temperature','top_p','max_tokens')},
                **({'reasoning_effort':self.config['reasoning_effort']} if 'reasoning_effort' in self.config else {}),
                'extra_body':{'thinking':{'type':self.config.get('thinking','disabled')}}}


class OpenAIClient(ModelClient):
    require_tool_call=True
    default_model=None

    def __init__(self,recorder,config):
        if not config.get('name'):raise ValueError('OpenAI models require an explicit model name')
        base_url=config.get('base_url','https://api.openai.com/v1')
        if 'gpt-oss' in config['name'] and base_url.rstrip('/')=='https://api.openai.com/v1':
            raise ValueError('gpt-oss requires a self-hosted or third-party base_url; it is not served by the OpenAI API')
        super().__init__(recorder,config,api_key_env=config.get('api_key_env','OPENAI_API_KEY'),base_url=base_url)

    def model_parameters(self):
        parameters={'max_completion_tokens':self.config['max_tokens']}
        if 'reasoning_effort' in self.config:parameters['reasoning_effort']=self.config['reasoning_effort']
        if self.config.get('reasoning_effort')=='none':
            parameters.update({k:self.config[k] for k in ('temperature','top_p') if k in self.config})
        return parameters


class OpenAICompatibleClient(OpenAIClient):
    def __init__(self,recorder,config):
        if not config.get('base_url'):raise ValueError('OpenAI-compatible models require an explicit base_url')
        super().__init__(recorder,config)

    def model_parameters(self):
        return {**{k:self.config[k] for k in ('temperature','top_p','max_tokens') if k in self.config},
                **({'reasoning_effort':self.config['reasoning_effort']} if 'reasoning_effort' in self.config else {}),
                **({'extra_body':self.config['extra_body']} if 'extra_body' in self.config else {})}


def create_model_client(recorder,config):
    provider=config.get('provider','deepseek')
    clients={'deepseek':DeepSeekClient,'openai':OpenAIClient,'openai_compatible':OpenAICompatibleClient}
    if provider not in clients:raise ValueError(f'Unknown model provider: {provider}')
    return clients[provider](recorder,config)
