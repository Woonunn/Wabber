"""Independent input-based checks for the three newly captured original prompts."""
import hashlib
import io
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd


def parameter_digest(parameters):
    h=hashlib.sha256()
    for p in parameters:
        x=p.detach().cpu().contiguous().numpy()
        h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
    return h.hexdigest()


def observe_training(seed):
    import torch
    from torch.optim.optimizer import register_optimizer_step_pre_hook,register_optimizer_step_post_hook
    torch.manual_seed(seed);torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.mkldnn.enabled=False
    records={}
    def parameters(opt):return [p for group in opt.param_groups for p in group['params']]
    def before(opt,args,kwargs):
        if id(opt) not in records:
            records[id(opt)]={'optimizer':type(opt).__name__,'steps':0,'initial_parameters_sha256':parameter_digest(parameters(opt))}
    def after(opt,args,kwargs):
        r=records[id(opt)];r['steps']+=1;r['final_parameters_sha256']=parameter_digest(parameters(opt))
    handles=[register_optimizer_step_pre_hook(before),register_optimizer_step_post_hook(after)]
    return records,handles


def items(value):
    result=json.loads(value)
    assert isinstance(result,list) and result and all(isinstance(x,str) for x in result),'Itemsets must be nonempty JSON string lists'
    assert len(set(result))==len(result),'Repeated item in itemset'
    return result


def check_mining(artifacts,basket):
    frequent=pd.read_csv(artifacts/'frequent_itemsets.csv');rules=pd.read_csv(artifacts/'association_rules.csv')
    assert len(frequent)>0 and len(rules)>0,'No actual frequent itemsets or association rules'
    observed=set()
    for row in frequent.itertuples():
        labels=items(row.itemsets);key=tuple(sorted(labels));assert key not in observed,'Duplicate itemset';observed.add(key)
        support=float(basket[labels].all(axis=1).mean())
        assert support>0 and np.isclose(support,row.support,rtol=1e-6,atol=1e-9),'Itemset support disagrees with original transactions'
    for row in rules.itertuples():
        a,b=items(row.antecedents),items(row.consequents)
        assert not set(a)&set(b),'Antecedent overlaps consequent'
        union=float(basket[a+b].all(axis=1).mean());ante=float(basket[a].all(axis=1).mean());cons=float(basket[b].all(axis=1).mean())
        assert ante>0 and cons>0 and np.isclose(row.support,union,rtol=1e-6,atol=1e-9)
        assert np.isclose(row.confidence,union/ante,rtol=1e-6,atol=1e-9),'Rule confidence disagrees with original baskets'
        assert np.isclose(row.lift,union/(ante*cons),rtol=1e-6,atol=1e-9),'Rule lift disagrees with original baskets'
    return {'transactions':len(basket),'frequent_itemsets':len(frequent),'association_rules':len(rules),'all_support_confidence_lift_recomputed':True}


def report(path,keywords,images=()):
    from pypdf import PdfReader
    from PIL import Image
    reader=PdfReader(path);text=' '.join(p.extract_text() or '' for p in reader.pages).lower()
    assert len(text)>300 and all(word in text for word in keywords),'Report is missing substantive task-specific text'
    if images:
        def pixels(im):
            rgb=im.convert('RGB');return (rgb.size,hashlib.sha256(rgb.tobytes()).hexdigest())
        embedded=set()
        for page in reader.pages:
            for image in page.images:
                with Image.open(io.BytesIO(image.data)) as im:embedded.add(pixels(im))
        for path in images:
            with Image.open(path) as im:assert pixels(im) in embedded,'Report omits an actual requested visualization'
    return {'pages':len(reader.pages),'text_characters':len(text)}


def credit(artifacts,inputs):
    raw=pd.read_excel(inputs/'default of credit card clients.xls',header=1).rename(columns={'default payment next month':'default'})
    assert len(raw)==30000
    discrete=raw[['SEX','EDUCATION','MARRIAGE','PAY_0','default']].astype(str)
    discrete['AGE']=pd.cut(raw.AGE,[0,30,40,50,150]).astype(str)
    discrete['LIMIT_BAL']=pd.cut(raw.LIMIT_BAL,[0,50000,150000,300000,np.inf]).astype(str)
    basket=pd.get_dummies(discrete,prefix_sep='=',dtype=bool);checks=check_mining(artifacts,basket)
    with (artifacts/'kmeans_checkpoint.pkl').open('rb') as f:saved=pickle.load(f)
    features=saved['features'];assert set(features)<=set(raw)-{'ID','default'}
    assert 'LIMIT_BAL' in features and any(c.startswith('BILL_AMT') for c in features) and any(c.startswith('PAY_') for c in features)
    scaler=saved['scaler'];x=scaler.transform(raw[features]);assert np.allclose(scaler.mean_,raw[features].mean())
    elbow=pd.read_csv(artifacts/'elbow_values.csv');assert elbow.k.tolist()==list(range(2,9))
    for row in elbow.itertuples():
        model=saved['models'][int(row.k)];assert model.n_clusters==row.k and model.n_iter_>0 and isinstance(model.random_state,int)
        labels=model.predict(x);actual=float(np.square(x-model.cluster_centers_[labels]).sum())
        assert np.isclose(actual,row.inertia,rtol=1e-6) and np.isclose(model.inertia_,actual,rtol=1e-6),'Elbow inertia disagrees with fitted centers and original clients'
    inertia=elbow.inertia.to_numpy();assert np.isfinite(inertia).all() and inertia[0]>inertia[-1] and (np.diff(inertia)<=1e-5).all()
    distances=1-(elbow.k.to_numpy()-2)/6-(inertia-inertia[-1])/(inertia[0]-inertia[-1])
    selected=int(elbow.k.iloc[int(np.argmax(distances))]);model=saved['models'][selected]
    clustered=pd.read_csv(artifacts/'kmeans_clusters.csv')
    assert np.array_equal(clustered.ID,raw.ID) and np.array_equal(clustered.cluster,model.predict(x)),'Persisted clusters do not match original clients and selected model'
    assert clustered.cluster.nunique()==selected
    checks.update(selected_k=selected,clients=len(raw),all_elbow_inertias_recomputed=True,
        pattern_report=report(artifacts/'pattern_analysis_report.pdf',['default']),
        segment_report=report(artifacts/'client_segments_analysis.pdf',['cluster','risk']))
    return checks


def retail(artifacts,inputs):
    sheets=pd.read_excel(inputs/'online_retail_II.xlsx',sheet_name=None);raw=pd.concat(sheets.values(),ignore_index=True)
    assert len(sheets)==2 and len(raw)==1067371
    d=raw[(raw.Quantity>0)&(raw.Price>0)&~raw.Invoice.astype(str).str.startswith('C')].copy();del raw,sheets
    d['InvoiceDate']=pd.to_datetime(d.InvoiceDate);d['revenue']=d.Quantity*d.Price;d['StockCode']=d.StockCode.astype(str)
    d['basket_id']=d.InvoiceDate.dt.year.astype(str)+':'+d.Invoice.astype(str)
    pairs=d[['basket_id','StockCode']].drop_duplicates();n=pairs.basket_id.nunique()
    eligible=pairs.StockCode.value_counts();eligible=eligible[eligible/n>=.03].index
    basket=pd.crosstab(pairs.loc[pairs.StockCode.isin(eligible),'basket_id'],pairs.loc[pairs.StockCode.isin(eligible),'StockCode']).reindex(pairs.basket_id.unique(),fill_value=0).gt(0)
    checks=check_mining(artifacts,basket)
    expected=d.groupby('StockCode').agg(quantity=('Quantity','sum'),revenue=('revenue','sum'),invoices=('basket_id','nunique')).sort_index()
    products=pd.read_csv(artifacts/'popular_products_analysis.csv',dtype={'StockCode':str}).set_index('StockCode').sort_index()
    assert products.index.equals(expected.index),'Product rows must preserve all original retained StockCode keys exactly'
    assert all(np.allclose(products[c],expected[c]) for c in ('quantity','revenue','invoices')),'Product totals do not match original transactions'
    countries=pd.read_csv(artifacts/'country_counts.csv').set_index('Country').transactions.sort_index();country=d.groupby('Country').size().sort_index()
    assert countries.index.equals(country.index) and np.array_equal(countries,country),'Country counts do not match transaction lines'
    monthly=pd.read_csv(artifacts/'monthly_sales.csv');monthly['Month']=pd.to_datetime(monthly.Month)
    actual=monthly.set_index('Month').revenue.sort_index();expected_month=d.set_index('InvoiceDate').revenue.resample('MS').sum()
    assert actual.index.equals(expected_month.index) and np.allclose(actual,expected_month),'Monthly revenue does not match original transactions'
    images=[artifacts/name for name in ('distribution_by_country.png','sales_trends_over_time.png','correlation_heatmap.png')]
    checks.update(source_rows=1067371,retained_rows=len(d),total_revenue=float(d.revenue.sum()),all_product_totals_recomputed=True,
        report=report(artifacts/'comprehensive_report.pdf',['country','sales','association'],images))
    return checks


def lstm(artifacts,inputs,evidence):
    import torch
    from torch import nn
    data=pd.read_csv(inputs/'data.csv');data['Date']=pd.to_datetime(data.Date)
    train=data.set_index('Date').Value1.loc['2023-01-01':'2023-05-31'].to_numpy(dtype=float)
    pred=pd.read_csv(artifacts/'predictions.csv');dates=pd.to_datetime(pred.Date)
    assert len(train)==151 and dates.tolist()==pd.date_range('2023-06-01','2023-07-19').tolist()
    values=pred.Predicted_Value1.to_numpy(dtype=float);assert np.isfinite(values).all()
    checkpoint=torch.load(artifacts/'model_checkpoint.pt',map_location='cpu',weights_only=True)
    class Predictor(nn.Module):
        def __init__(self):
            super().__init__();self.lstm=nn.LSTM(1,int(checkpoint['hidden_size']),num_layers=int(checkpoint['num_layers']),batch_first=True)
            self.fc=nn.Linear(int(checkpoint['hidden_size']),1)
        def forward(self,x):return self.fc(self.lstm(x)[0][:,-1,:])
    model=Predictor();model.load_state_dict(checkpoint['state_dict']);model.eval()
    mean,scale=float(checkpoint['mean']),float(checkpoint['scale']);lookback=int(checkpoint['lookback'])
    assert 1<=lookback<len(train) and scale>0 and np.isclose(mean,train.mean(),atol=1e-5) and np.isclose(scale,train.std(),atol=1e-5),'Scaler must use only Jan-May Value1'
    window=np.asarray(checkpoint['last_window'],dtype=np.float32)
    assert np.allclose(window,((train-mean)/scale)[-lookback:],atol=1e-5),'Forecast window differs from original final training observations'
    result=[];window=torch.tensor(window,dtype=torch.float32)
    with torch.no_grad():
        for _ in range(49):
            value=model(window[None,:,None]).reshape(());result.append(float(value)*scale+mean);window=torch.cat([window[1:],value.reshape(1)])
    assert np.allclose(values,result,rtol=1e-5,atol=1e-6),'Saved forecasts are not reproducible from the actual LSTM checkpoint'
    records=evidence['model_checkpoint.pt']['optimizer_steps']
    matches=[r for r in records if r['steps']>0 and r['initial_parameters_sha256']!=r['final_parameters_sha256'] and r['final_parameters_sha256']==parameter_digest(model.parameters())]
    assert matches,'No observed real optimizer updates leading to the saved LSTM weights'
    summary=json.loads((artifacts/'training_summary.json').read_text());assert summary['train_rows']==151 and summary['training_start']=='2023-01-01' and summary['training_end']=='2023-05-31'
    assert summary['epochs_run']>0 and np.isfinite(summary['initial_loss']) and np.isfinite(summary['final_loss']) and summary['final_loss']<summary['initial_loss']
    actual=data.set_index('Date').Value1.reindex(pd.date_range('2023-06-01','2023-07-19')).to_numpy()
    return {'training_rows':151,'forecast_rows':49,'optimizer_steps_observed':sum(r['steps'] for r in matches),
        'checkpoint_reproduces_forecasts':True,'forecast_rmse':float(np.sqrt(np.mean((values-actual)**2))),
        'last_value_baseline_rmse':float(np.sqrt(np.mean((train[-1]-actual)**2))),'official_gt_mae_checked':False}


def validate(task,artifacts,inputs,evidence):
    if task=='human_142':return credit(artifacts,inputs)
    if task=='human_7':return retail(artifacts,inputs)
    if task=='csv_excel_29':return lstm(artifacts,inputs,evidence)
    raise ValueError(task)
