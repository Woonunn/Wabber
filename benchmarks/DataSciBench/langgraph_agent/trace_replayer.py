import asyncio
import json
from pathlib import Path
from atsuite.tracing.storage import read_text


class ReplayModel:
    def __init__(self,capture,recorder,normalizer=None,calibration=False,*,input_equal=None):
        self.input_equal=input_equal or (lambda left,right:left==right)
        self.calibration=calibration
        self.normalizer=normalizer
        if json.loads(read_text(Path(capture)/'result.json')).get('status')!='complete':
            raise ValueError('Only successful Capture traces can be replayed')
        self.events=[json.loads(line) for line in read_text(Path(capture)/'events.jsonl').splitlines()]
        self.llm=[e for e in self.events if e['kind']=='llm'];self.index=0;self.recorder=recorder

    async def decide(self,phase,iteration,messages):
        if self.index>=len(self.llm):raise ValueError('Replay requested an uncaptured LLM decision')
        event=self.llm[self.index];self.index+=1
        if (event['phase'],event['iteration'])!=(phase,iteration):raise ValueError('LLM control flow diverged from Capture')
        def input_view(items):
            if self.normalizer is None:return items
            result=[]
            for item in items:
                try:content=json.loads(item['content'])
                except (ValueError,TypeError):result.append(item)
                else:result.append({**item,'content':self.normalizer(content)})
            return result
        if not self.calibration and not self.input_equal(input_view(messages),input_view(event['request']['messages'])):raise ValueError('Replay LLM inputs diverged from Capture')
        if self.calibration:
            event={**event,'request':{**event['request'],'messages':messages}}
        if messages!=event['request']['messages']:self.recorder.emit('replay_input_equivalence',phase=phase,iteration=iteration,messages=messages,basis='sealed calibrated rendering digests; original model decision unchanged')
        self.recorder.emit('llm_start',phase=phase,iteration=iteration,request=event['request'])
        # Recorded model latency is an explicit replay wait, never CPU computation.
        await asyncio.sleep(event['duration_s'])
        self.recorder.emit('llm',**{k:v for k,v in event.items() if k not in {'seq','kind','utc','monotonic_ns','request_id','session_id','source'}},source='environment_reference_frozen_decision' if self.calibration else 'replayed_capture_no_api_charge')
        return event['decision']

    def assert_consumed(self):
        if self.index!=len(self.llm):raise ValueError('Replay did not consume all captured LLM decisions')
