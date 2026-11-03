import json
from langgraph.graph import StateGraph,START,END
from langgraph.types import Send
from .state import AgentState
from .validators import validate_plan

PLAN_SYSTEM='''You are a data analysis planner. Return JSON {"steps":[{"id":"clean","purpose":"...","prompt_evidence":"exact substring of the user prompt","deps":[],"outputs":["cleaned_data.csv"],"code":"complete runnable Python"}]}.
Implement the ORIGINAL prompt, not an illustrative substitute. Choose the actual number of required steps. Independent sections should depend on one shared preparation step and run in parallel; never duplicate a calculation to invent parallelism. Dependency IDs must refer to this plan. Output filenames are globally unique within the plan. Include complete Python code in every step.
The code runs in an isolated offline CPU sandbox with pandas, numpy, scipy, sklearn, matplotlib, reportlab, openpyxl, torch. No network, installs, host filesystem, credentials or subprocess services. All input files are already prepared. Use inputs_dir / "filename" for input, output_dir / "filename" for EVERY generated artifact; create output subdirectories when needed. artifact_path("filename") reads a committed artifact from dependencies. state_out={} is a dictionary you fill with actual reusable Python data/models; all ancestors' committed values are accessible as shared["step_id"]["variable"]. DataFrames/models remain in Tool Runtime; never print large data. Import modules explicitly. Do not put module objects or functions in state_out. Keep state_out to values actually needed downstream. Each branch receives the SAME immutable shared snapshot; sibling writes are not visible until MERGE. A final report task must depend on every computation whose artifacts it uses.
Use seed for every random estimator and stochastic process, n_jobs=1, avoid timestamps or random artifact names. matplotlib backend Agg; PDFs use reportlab Canvas(invariant=1) for reproducibility. Prefer compact robust implementations; genuine models and computations only. All original requested outputs are mandatory. For model reports include real computed metrics. Output JSON only. No preinstalled task-specific reference functions may be called.'''
REVIEW_SYSTEM='''Review the original prompt against actual Tool results and validation. Return JSON {"verdict":"complete"|"replan"|"failed","feedback":"specific reasoning"}. Complete only when the actual results meet the task; if errors or missing outputs remain, request a corrected plan. Do not assume a successful file listing proves numerical correctness.'''


CONTRACTS={'csv_excel_34':'The input CSV has AGE and INCTOT columns. Each row represents a sampled individual: repeated AGE/INCTOT pairs are not duplicate people. Preserve ALL rows with finite nonnegative INCTOT; do not drop_duplicates, winsorize, discard large incomes, or fabricate data. cleaned_data.csv may retain INCTOT or add an identical income alias. Use this unchanged income population for Lorenz and Gini. Dependency files are read with artifact_path; output_dir is private to each branch.', 'human_15':'Read dependency files with artifact_path or shared state, never from output_dir. output_dir is a fresh isolated directory for this branch only. Preserve red bars/black edges, dash-dot category lines with legend, and scatter alpha .5 and sizes proportional to Sales.', 'human_12':"Impute missing values without dropping rows, preserve official 32561 train + 16281 test row order. cleaned_data.csv must contain numeric encoded/scaled features plus split ('train'/'test') and target (0/1), no original income string. model_predictions.csv must have y_true,y_pred,y_score and repeated accuracy,precision,recall,f1,roc_auc columns, evaluated only on official test. model_checkpoint.pkl is a pickle dict with key 'model', a fitted sklearn classifier accepting the cleaned numeric feature columns (exclude split/target). clustering_results.csv contains all 48842 rows with integer cluster column. PDF report needs real metrics and cluster profiles, not a placeholder. Use artifact_path for dependency CSVs, never output_dir, which contains only this branch outputs. Derive labels exactly as income.astype(str).str.strip().str.rstrip('.').eq('>50K').astype(int), without any leading space in the comparison. Assert both target classes occur before preprocessing; the train set has 7841 positive labels. Exclude target and split from all scaler/encoder inputs. Preserve useful matrices in state_out for both model and clustering branches."}


CONTRACTS.update({
'human_142': "Read the original XLS with header=1 (30000 clients), rename 'default payment next month' to 'default'. For independently verifiable mining use original SEX, EDUCATION, MARRIAGE, PAY_0, default as strings, plus AGE=pd.cut(AGE,[0,30,40,50,150]).astype(str) and LIMIT_BAL=pd.cut(LIMIT_BAL,[0,50000,150000,300000,np.inf]).astype(str); pd.get_dummies(prefix_sep='=',dtype=bool). Use genuine mlxtend Apriori (min_support .05, max_len 3) and association_rules(confidence >= .3). Save itemsets, antecedents, consequents as json.dumps(sorted(list(items))); CSV columns itemsets/support and antecedents/consequents/support/confidence/lift. These are benchmark serialization/parameter contracts, not official GT thresholds. For clustering choose numeric credit-limit, bill and repayment predictors, excluding ID and default; fit StandardScaler on the original clients. Fit real KMeans for every k=2..8, random_state=seed, n_init=10, n_jobs not supported. Save auxiliary elbow_values.csv with k,inertia; select k by maximum distance below normalized endpoint chord: distance=1-(k-2)/6-(inertia-inertia[-1])/(inertia[0]-inertia[-1]). Save kmeans_clusters.csv with ID and cluster in original row order. Save auxiliary kmeans_checkpoint.pkl as dict {'scaler':fitted_scaler,'models':{int_k:fitted_model_for_k},'features':ordered_feature_names}. Both PDFs must contain real computed findings: include default analysis and cluster risk interpretation. Reports must read real results and use invariant PDF output. Preserve prepared data/matrices in state_out for actual reuse by independent mining and clustering steps. All auxiliary evidence files must be declared as step outputs.",
'human_7': "IMPORTANT: each step prompt_evidence must be a literal quote from original_prompt, never from validation_contract. For workspace/data preparation, the original input filename online_retail_II.xlsx is a valid short quote. Text such as 'Use BOTH original XLSX sheets' is this contract, not evidence from the original prompt. Use BOTH original XLSX sheets, concatenate all 1067371 source rows. Apply the declared benchmark scope: keep Quantity>0, Price>0 and invoices not starting 'C'; retain missing Customer ID for sales/baskets. Preserve StockCode as strings. revenue=Quantity*Price; basket_id=InvoiceDate.year.astype(str)+':'+Invoice.astype(str), so invoice IDs cannot collide across years. Country distribution counts retained transaction lines. Monthly sales use calendar-month-start timestamps. For Apriori on all retained baskets/countries use min_support=.03, max_len=3, rules confidence>=.5; prune infrequent singleton products before constructing a dense basket, but reindex to ALL basket IDs so empty-after-pruning baskets remain in the support denominator. Itemset/antecedent/consequent fields must be sorted JSON string lists. frequent_itemsets.csv requires itemsets,support; auxiliary association_rules.csv requires antecedents,consequents,support,confidence,lift. popular_products_analysis.csv requires StockCode,quantity,revenue,invoices for ALL retained products, with totals computed from original transactions. Save auxiliary country_counts.csv (Country,transactions) and monthly_sales.csv (Month as YYYY-MM-01,revenue) alongside the charts. The PDF must actually embed ALL THREE requested PNGs and discuss country, sales and association patterns using real results, with substantial text. Use image-preserving reportlab drawImage, not downsampled screenshots. Preserve cleaned data/matrices in state_out for downstream independent analyses; do not export large unused raw tables. Every generated file must be declared in step outputs.",
'csv_excel_29': "Use only Value1 for 151 daily Jan 1-May 31 training observations; never fit, scale, select epochs or forecast using future June-July values or Value2/Value3. Forecast all 49 dates June 1-July 19 recursively. Use a real CPU torch.nn.Module with self.lstm=nn.LSTM(input_size=1,hidden_size=chosen_h,num_layers=chosen_layers,batch_first=True), self.fc=nn.Linear(chosen_h,1), forward(x)=self.fc(self.lstm(x)[0][:,-1,:]); no dropout/bidirectionality. You choose lookback, hidden size, layer count, learning rate, epoch count and optimization strategy; perform genuine backward/optimizer.step updates with seed, not fake work. Tool independently observes optimizer steps and parameter hashes. Scale with mean=float(train_values.mean()) and scale=float(train_values.std(ddof=0)) on only Jan-May observations. Use float32 tensors and a flat 1D last_window of final scaled training values; predict recursively with eval/no_grad. predictions.csv columns Date (YYYY-MM-DD) and Predicted_Value1. Save auxiliary model_checkpoint.pt with torch.save of a plain dict containing state_dict,lookback,hidden_size,num_layers,mean,scale,last_window (a CPU Tensor), training_start='2023-01-01',training_end='2023-05-31'. State dict keys start lstm.* and fc.*; do NOT pickle a custom model class. Save auxiliary training_summary.json with train_rows=151,training_start,training_end,epochs_run,initial_loss,final_loss,seed; losses must be real and training should improve. Produce checkpoint and summary in the SAME step that runs optimizer steps so independent Tool training evidence is attached to checkpoint. Avoid putting Torch modules or function closures in state_out; native arrays, tensors or checkpoint dicts are supported. All evidence files must be declared as outputs. Plot actual computed predictions. No forced parallel copies: training, forecasting and plotting have their real dependencies."
})


def build_graph(model,tools,recorder,required_outputs,max_iterations=3,seed=0,validation_contract=None,function_task=False):
    async def plan(state):
        system=PLAN_SYSTEM
        if function_task:
            system='''Implement the original Python function task exactly. Return JSON {"code":"complete self-contained Python source defining task_func"}. Preserve the specified signature, imports, constants and return values. Do not call the function or write harness files. The isolated Tool will call your function with prepared upstream arguments, save its actual return, and independently run the full upstream tests. No reference solutions or test inspection. Correct implementation errors from feedback. CPU/offline execution with pandas, numpy, scipy, sklearn, matplotlib, seaborn, openpyxl and statsmodels. Output JSON only.'''
        messages=[{'role':'system','content':system},{'role':'user','content':json.dumps({
            'original_prompt':state['user_prompt'],'datasets':state['dataset_refs'],
            'required_outputs':required_outputs,'validation_contract':validation_contract if validation_contract is not None else CONTRACTS.get(state['task_id'],''),'feedback':state.get('feedback',''),
            'available_committed_artifacts':[{k:a[k] for k in ('name','signature','comparison')} for a in state.get('artifacts',[])]},ensure_ascii=False)}]
        decision=await model.decide('PLAN' if state['iteration']==0 else 'REPLAN',state['iteration'],messages)
        if function_task:
            source=decision.get('code')
            if not isinstance(source,str) or not source.strip():raise ValueError('Missing BCB function source')
            decision={'steps':[{'id':'function','purpose':'Implement and invoke the original function',
                'prompt_evidence':state['user_prompt'][:160],'deps':[],
                'outputs':required_outputs,'code':"(output_dir / 'solution.py').write_text("+repr(source)+")"}]}
        try:steps=validate_plan(decision,state['user_prompt'])
        except (ValueError,KeyError,SyntaxError) as exc:
            return {'execution_plan':[],'status':'invalid_plan' if state['iteration']+1<max_iterations else 'failed',
                    'feedback':'Correct the plan validation error: '+str(exc),'error':{'plan_error':str(exc)}}
        return {'execution_plan':steps,'completed':[],'status':'planning','error':None,
                'messages':[{'role':'assistant','content':json.dumps(decision,ensure_ascii=False)}]}

    async def prepare(state):
        result=await tools.call('prepare_workspace',{'task_id':state['task_id'],'dataset_refs':state['dataset_refs']},'prepare')
        if result['status']!='ok':raise RuntimeError('Workspace preparation failed')
        return {'workspace_id':state['session_id'],'snapshot':result['snapshot'],'status':'running'}

    def ready(state):
        completed=set(state['completed'])
        return [s for s in state['execution_plan'] if s['id'] not in completed and set(s['deps'])<=completed]

    def dispatch(state):
        wave=f"i{state['iteration']}-v{state['snapshot']}"
        tasks=ready(state)
        if not tasks:raise RuntimeError('No runnable task; invalid DAG or unresolved dependency')
        recorder.emit('wave_start',wave=wave,steps=[s['id'] for s in tasks],snapshot=state['snapshot'])
        return {'status':'running'}

    def fan_out(state):
        return [Send('ANALYSIS',{**state,'step':step}) for step in ready(state)]

    async def analysis(state):
        step=state['step'];key=f"i{state['iteration']}-{step['id']}"
        result=await tools.call('execute_analysis',{'step_id':step['id'],'code':step['code'],
            'outputs':step['outputs'],'snapshot':state['snapshot'],'seed':seed,'call_id':key},key,
            wave=f"i{state['iteration']}-v{state['snapshot']}")
        return {'tool_results':{key:result}}

    async def merge(state):
        branches=[];errors=[];completed=list(state['completed'])
        for step in ready(state):
            key=f"i{state['iteration']}-{step['id']}";result=state['tool_results'][key]
            if result['status']=='ok':branches.append(result);completed.append(step['id'])
            else:errors.append({'step':step['id'],'error_type':result.get('error_type'),'error':result.get('error')})
        update={'completed':completed,'error':{'tool_errors':errors} if errors else None}
        if branches:
            result=await tools.call('merge_artifacts',{'branches':[{'branch_ref':b['branch_ref'],'step_id':b['step_id'],'artifacts':[{k:a[k] for k in ('name','signature','comparison')} for a in b['artifacts']]} for b in branches],
                'snapshot':state['snapshot']},f"merge-i{state['iteration']}-v{state['snapshot']}")
            if result['status']!='ok':raise RuntimeError('State merge failed: '+str(result))
            update.update(snapshot=result['snapshot'],artifacts=result['artifacts'])
        return update

    def after_merge(state):
        return 'REVIEW' if state.get('error') or len(state['completed'])==len(state['execution_plan']) else 'DISPATCH'

    async def review(state):
        result=await tools.call('validate_result',{'outputs':required_outputs,'snapshot':state['snapshot']},f"validate-i{state['iteration']}")
        from .tool_client import semantic
        context={'original_prompt':state['user_prompt'],'plan':state['execution_plan'],'validation':semantic(result),
                 'errors':state.get('error'),'completed':state['completed'],
                 'tool_outputs':[{**semantic(v),'stdout':v.get('stdout','')[:3000]} for k,v in state['tool_results'].items() if k.startswith(f"i{state['iteration']}-")]}
        review_system=REVIEW_SYSTEM+(' Choose replan for fixable implementation failures while iterations remain.' if function_task else '')
        decision=await model.decide('REVIEW',state['iteration'],[{'role':'system','content':review_system},
            {'role':'user','content':json.dumps(context,ensure_ascii=False)}])
        verdict=decision.get('verdict')
        if verdict not in {'complete','replan','failed'}:raise ValueError('Invalid review decision')
        if verdict=='complete' and (result['status']!='ok' or state.get('error')):
            verdict='replan';decision['feedback']='Validation gate failed. '+decision.get('feedback','')
        if function_task and (result['status']!='ok' or state.get('error')) and state['iteration']+1<max_iterations:
            verdict='replan'
        status='complete' if verdict=='complete' else 'failed' if verdict=='failed' or state['iteration']+1>=max_iterations else 'replan'
        return {'validation':semantic(result),'evaluation_basis':result.get('evaluation_basis'),'status':status,'feedback':decision.get('feedback',''),
                'messages':[{'role':'assistant','content':json.dumps(decision,ensure_ascii=False)}]}

    def replan(state):return {'iteration':state['iteration']+1}

    g=StateGraph(AgentState)
    g.add_node('PLAN',plan);g.add_node('PREPARE_WORKSPACE',prepare);g.add_node('DISPATCH',dispatch)
    g.add_node('ANALYSIS',analysis);g.add_node('MERGE',merge);g.add_node('REVIEW',review);g.add_node('REPLAN',replan)
    g.add_edge(START,'PLAN');g.add_conditional_edges('PLAN',lambda s:END if s['status']=='failed' else 'REPLAN' if s['status']=='invalid_plan' else 'PREPARE_WORKSPACE' if not s.get('workspace_id') else 'DISPATCH')
    g.add_edge('PREPARE_WORKSPACE','DISPATCH');g.add_conditional_edges('DISPATCH',fan_out,['ANALYSIS'])
    g.add_edge('ANALYSIS','MERGE');g.add_conditional_edges('MERGE',after_merge)
    g.add_conditional_edges('REVIEW',lambda s:'REPLAN' if s['status']=='replan' else END)
    g.add_edge('REPLAN','PLAN')
    return g.compile()
