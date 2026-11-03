"""Independent task checks; imported and run only inside the Tool sandbox."""
import json,re
from pathlib import Path
import numpy as np
import pandas as pd


def validate(task,artifacts,inputs,evidence):
    checks={}
    if task.startswith('bcb'):
        from bcb_checks import validate as validate_bcb
        return validate_bcb(task,artifacts,inputs)
    elif task=='csv_excel_34':
        clean=pd.read_csv(artifacts/'cleaned_data.csv');columns=[c for c in clean if 'inc' in c.lower()]
        if not columns:raise ValueError('Missing income column in cleaned data')
        values=clean[columns[0]].to_numpy(dtype=float)
        if not all(np.allclose(values,clean[c].to_numpy(dtype=float),equal_nan=True) for c in columns[1:]):
            raise ValueError('Conflicting income columns in cleaned data')
        x=np.sort(values)
        source=pd.read_csv(inputs/'data_pums_2000.csv').INCTOT.to_numpy(dtype=float)
        expected=np.sort(source[np.isfinite(source)&(source>=0)])
        assert len(x)==len(expected) and np.allclose(x,expected),'Cleaning changed the population: preserve all finite nonnegative income rows, including repeated attribute pairs'
        assert len(x)>1000 and np.isfinite(x).all() and x.min()>=0 and x.sum()>0,'Invalid cleaned income data'
        g=2*np.dot(np.arange(1,len(x)+1),x)/(len(x)*x.sum())-(len(x)+1)/len(x)
        saved=float(re.search(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?',(artifacts/'gini_coefficient.txt').read_text())[0])
        assert np.isclose(saved,g,atol=1e-6),'Gini does not match actual cleaned observations'
        checks.update(rows=len(x),gini=float(g),gini_independently_verified=True)
    elif task=='human_15':
        bar=evidence['bar_chart.png'];line=evidence['line_chart.png'];scatter=evidence['scatter_chart.png']
        patches=bar['bars'];assert sorted(round(p['height'],6) for p in patches)==[4800,7800],'Incorrect annual sales bars'
        assert all(np.allclose(p['face'],[1,0,0,1]) and np.allclose(p['edge'],[0,0,0,1]) for p in patches),'Wrong bar colors'
        lines=line['lines'];assert len(lines)==3 and all(p['style']=='-.' for p in lines),'Wrong category line styles'
        assert sorted(round(sum(v for v in p['y'] if v is not None),6) for p in lines)==[2100,4800,5700],'Incorrect category totals'
        collections=scatter['scatter'];ys=[];ratios=[];colors=[]
        for p in collections:
            ys+=p['y'];ratios.extend(np.array(p['sizes'])/np.array(p['y']));colors.extend(p['face'])
            assert p['alpha']==.5,'Wrong scatter alpha'
        assert sorted(round(x,6) for x in ys)==list(range(1500,2200,100)),'Incorrect scatter observations'
        assert np.allclose(ratios,ratios[0]) and len({str(c) for c in colors})==3,'Incorrect point sizes or category colors'
        checks.update(year_totals={'2022':4800,'2023':7800},chart_geometry_verified=True)
    elif task=='human_12':
        from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
        import pickle
        clean=pd.read_csv(artifacts/'cleaned_data.csv');pred=pd.read_csv(artifacts/'model_predictions.csv')
        assert set(['split','target'])<=set(clean),'Required cleaned data contract: split,target'
        test=clean[clean['split']=='test'];train=clean[clean['split']=='train']
        assert len(train)==32561 and len(test)==16281,'Official train/test split was not preserved'
        labels=[]
        for name in ('adult.data','adult.test'):
            raw=pd.read_csv(inputs/name,header=None,skipinitialspace=True,comment='|')
            labels.extend((raw.iloc[:,-1].str.rstrip('.')=='>50K').astype(int).tolist())
        assert np.array_equal(clean.target,labels),'Cleaned targets differ from official data'
        assert len(pred)==len(test) and np.array_equal(pred.y_true,test.target),'Holdout target alignment error'
        scores={'accuracy':accuracy_score(pred.y_true,pred.y_pred),'precision':precision_score(pred.y_true,pred.y_pred),
                'recall':recall_score(pred.y_true,pred.y_pred),'f1':f1_score(pred.y_true,pred.y_pred),
                'roc_auc':roc_auc_score(pred.y_true,pred.y_score)}
        assert scores['accuracy']>=.8,'Holdout accuracy below .80'
        assert all(k in pred and np.allclose(pred[k],v) for k,v in scores.items()),'Saved metrics fail independent recomputation'
        with (artifacts/'model_checkpoint.pkl').open('rb') as f:saved=pickle.load(f)
        model=saved['model'];x=test.drop(columns=['split','target'])
        assert np.array_equal(model.predict(x),pred.y_pred),'Reloaded model does not reproduce predictions'
        clusters=pd.read_csv(artifacts/'clustering_results.csv')
        assert len(clusters)==len(clean) and 'cluster' in clusters and clusters.cluster.nunique()>=2,'Invalid clustering output'
        from pypdf import PdfReader
        text=' '.join(p.extract_text() or '' for p in PdfReader(artifacts/'analysis_report.pdf').pages).lower()
        assert len(text)>500 and 'accuracy' in text and 'cluster' in text,'Insufficient report content'
        checks.update(heldout_metrics=scores,checkpoint_reloaded=True,cluster_count=int(clusters.cluster.nunique()))
    elif task in ('human_142','human_7','csv_excel_29'):
        from extended_checks import validate as extended_validate
        return extended_validate(task,artifacts,inputs,evidence)
    else:raise ValueError('No independent result validator configured for task '+task)
    return checks
