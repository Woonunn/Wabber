"""Batch AWS log reads and retain only measured, uniquely identified usage records."""
import json
import math
from atsuite.aws.CloudWatch import AWSCloudWatch


def session_usage(events, session_id):
    rows=[];seen=set()
    for event in events:
        try:payload=json.loads(event.get('message',''))
        except (ValueError,TypeError):continue
        if payload.get('attributes',{}).get('session.id')!=session_id:continue
        stamp=payload.get('event_timestamp') or payload.get('timestamp') or event.get('timestamp')
        key=json.dumps([stamp,payload],sort_keys=True) if stamp is not None else event.get('eventId') or json.dumps(payload,sort_keys=True)
        if key in seen:continue
        seen.add(key);rows.append(payload)
    fields={'vcpu_hours':'agent.runtime.vcpu.hours.used','memory_gb_hours':'agent.runtime.memory.gb_hours.used'}
    result={'session_id':session_id,'log_entries':len(rows),'vcpu_hours':None,'memory_gb_hours':None,'raw_events':rows}
    for name,key in fields.items():
        values=[r.get('metrics',{}).get(key) for r in rows]
        if values and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0 for v in values):result[name]=sum(values)
    elapsed=[r.get('attributes',{}).get('time_elapsed_seconds') for r in rows]
    def valid(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0
    stamps=[r.get('event_timestamp') for r in rows]
    result['residence']={
        'reported_usage_seconds':sum(elapsed) if elapsed and all(valid(v) for v in elapsed) else None,
        'first_usage_epoch_ms':min(stamps) if stamps and all(valid(v) for v in stamps) else None,
        'last_usage_epoch_ms':max(stamps) if stamps and all(valid(v) for v in stamps) else None,
        'complete':False,
        'scope':'observed session usage only; no creation or termination evidence',
    }
    return result


class BatchCloudWatch(AWSCloudWatch):
    def __init__(self,region):
        super().__init__(region=region);self._batch_logs={};self._usage_groups={}

    def get_logs(self,resource_type,resource_name,start_time,end_time,request_id=None):
        key=(resource_type,resource_name,start_time,end_time)
        if key not in self._batch_logs:self._batch_logs[key]=super().get_logs(resource_type,resource_name,start_time,end_time,request_id=None)
        events=self._batch_logs[key]
        return [e for e in events if request_id in e.get('message','')] if request_id else events

    def get_agentcore_usage_from_logs(self,runtime_id,session_id,start_time,end_time):
        key=(runtime_id,start_time,end_time)
        if key not in self._usage_groups:
            groups=self.discover_agentcore_usage_log_groups(runtime_id)
            if not groups:raise RuntimeError('No AgentCore usage log group: '+runtime_id)
            events=[]
            for group in groups:events.extend(self._filter_log_group_events(group,start_time,end_time))
            # Index once: a long run must not reparse every other Session's usage.
            by_session={}
            for event in events:
                try:payload=json.loads(event.get('message',''))
                except (ValueError,TypeError):continue
                sid=payload.get('attributes',{}).get('session.id')
                if sid:by_session.setdefault(sid,[]).append(event)
            self._usage_groups[key]=by_session
        return session_usage(self._usage_groups[key].get(session_id,[]),session_id)


def collect_ali(context, invocations):
    from atsuite.ali.ali import Ali
    from atsuite.ali.sls import AliSLS
    from atsuite.analysis.model import CollectionResult,ProviderMetric,EvidenceRecord
    from alibabacloud_fc20230330.models import GetFunctionRequest
    from alibabacloud_sls20201230.models import GetLogsRequest
    from alibabacloud_tea_util.models import RuntimeOptions
    result=CollectionResult();name=next(iter(context.targets.values()))['name']
    function=Ali().get_fc_client().get_function(name,GetFunctionRequest()).body
    config=function.log_config
    if not config or not config.project or not config.logstore:raise RuntimeError('No configured FC request log destination')
    sls=AliSLS(project=config.project,location=context.targets[next(iter(context.targets))].get('region','us-east-1'))
    columns=['requestId','durationMs','memoryUsageMB','isColdStart','coldStartLatencyMs','scheduleLatencyMs','invokeFunctionStartTimestamp','instanceId']
    query='* | SELECT '+', '.join(columns)+" FROM log WHERE operation = 'InvokeFunction' LIMIT 100000"
    response=sls.client.get_logs_with_options(config.project,config.logstore,GetLogsRequest(from_=int(context.start_time)-300,to=int(context.end_time)+300,query=query),{},RuntimeOptions())
    raw=response.body or []
    if len(raw)>=100000:raise RuntimeError('SLS result may be truncated; do not aggregate an incomplete window')
    result.evidence.append(EvidenceRecord(evidence_id='ali_sls:window',provider=context.provider,source='sls',query={'project':config.project,'logstore':config.logstore,'query':query},raw=raw))
    rows={}
    for entry in raw:
        row=json.loads(entry['_col0']) if '_col0' in entry else entry
        if isinstance(row,list):row=dict(zip(columns,row))
        row={k.lower():v for k,v in row.items()}
        if row.get('requestid'):rows[str(row['requestid'])]=row
    names={'duration_ms':'durationms','memory_usage_mb':'memoryusagemb','is_cold_start':'iscoldstart','cold_start_latency_ms':'coldstartlatencyms','schedule_latency_ms':'schedulelatencyms','instance_id':'instanceid','invoke_start_epoch_ms':'invokefunctionstarttimestamp'}
    for invocation in invocations:
        row=rows.get(invocation.provider_request_id,{})
        fields={k:row.get(v) for k,v in names.items()}
        result.metrics.append(ProviderMetric(metric_id=invocation.call_id,provider=context.provider,source='sls',invocation_call_id=invocation.call_id,provider_request_id=invocation.provider_request_id,fields=fields,evidence_refs=['ali_sls:window']))
        if not row:result.diagnostics.append({'kind':'unmatched_request_id','provider_request_id':invocation.provider_request_id,'call_id':invocation.call_id})
    return result
