"""
save_plots.py — Export every app plot to PNG using default slider settings.
Run:  python save_plots.py
Output goes into  ./saved_plots/
"""

import os, math, numpy as np, pandas as pd
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from app_data import (
    df_model, df_raw_pct, LAG1_FEATURE_COLS, INDEX_NAMES,
    friendly_name, get_Xy, make_confusion_fig, make_roc_fig,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_plots")
os.makedirs(OUT, exist_ok=True)

def save(fig, name, w=1200, h=None):
    path = os.path.join(OUT, f"{name}.png")
    kw = dict(width=w, scale=2)
    if h:
        kw["height"] = h
    fig.write_image(path, **kw)
    print(f"  ✓ {name}.png")


# ── defaults (same as the slider initial values) ──
DT_DEPTH   = 3
DT_TEST    = 0.20
RF_TREES   = 100
RF_DEPTH   = 4
RF_TEST    = 0.20
GB_TREES   = 100
GB_DEPTH   = 3
GB_LR      = 0.10
GB_TEST    = 0.20
CMP_TEST   = 0.20

feat_names = LAG1_FEATURE_COLS

# ===================================================================
#  1.  DATA OVERVIEW — Class Balance
# ===================================================================
print("\n─── Data Overview ───")
up = int(df_model['NIFTY_Direction'].sum())
down = len(df_model) - up
n = len(df_model)

fig = go.Figure()
fig.add_trace(go.Bar(x=['DOWN ↓'], y=[down], marker_color='#ef4444',
                     text=[f'{down}<br>({down/n*100:.1f}%)'],
                     textposition='inside', textfont=dict(size=16, color='white')))
fig.add_trace(go.Bar(x=['UP ↑'], y=[up], marker_color='#22c55e',
                     text=[f'{up}<br>({up/n*100:.1f}%)'],
                     textposition='inside', textfont=dict(size=16, color='white')))
fig.update_layout(showlegend=False, height=350, yaxis_title='Number of Trading Days',
                  margin=dict(l=50, r=20, t=20, b=40))
save(fig, "01_class_balance", h=350)

# ===================================================================
#  2.  DATA OVERVIEW — Conditional Bars
# ===================================================================
rows = []
for col in LAG1_FEATURE_COLS:
    name = friendly_name(col).replace(' (prev day)', '')
    for direction, label in [(1, 'UP yesterday'), (0, 'DOWN yesterday')]:
        mask = df_model[col] == direction
        n_total = mask.sum()
        n_nifty_up = df_model.loc[mask, 'NIFTY_Direction'].sum()
        pct_up = (n_nifty_up / n_total * 100) if n_total > 0 else 0
        rows.append({'Exchange': name, 'Condition': label, 'NIFTY UP %': pct_up, 'Count': n_total,
                     'NIFTY UP': int(n_nifty_up), 'NIFTY DOWN': int(n_total - n_nifty_up)})
df_bars = pd.DataFrame(rows)
fig = go.Figure()
up_data = df_bars[df_bars['Condition'] == 'UP yesterday']
fig.add_trace(go.Bar(x=up_data['Exchange'], y=up_data['NIFTY UP %'],
                     name='Exchange was UP ↑ yesterday', marker_color='#22c55e',
                     text=[f"{v:.0f}%" for v in up_data['NIFTY UP %']], textposition='outside'))
down_data = df_bars[df_bars['Condition'] == 'DOWN yesterday']
fig.add_trace(go.Bar(x=down_data['Exchange'], y=down_data['NIFTY UP %'],
                     name='Exchange was DOWN ↓ yesterday', marker_color='#ef4444',
                     text=[f"{v:.0f}%" for v in down_data['NIFTY UP %']], textposition='outside'))
fig.add_hline(y=50, line_dash='dash', line_color='#94a3b8', line_width=1,
              annotation_text='50% (coin flip)', annotation_position='top left')
fig.update_layout(barmode='group', height=620, yaxis_title='% of days NIFTY went UP',
                  yaxis_range=[0, 78], xaxis_tickangle=-25,
                  legend=dict(orientation='h', y=1.06, x=0.5, xanchor='center'),
                  margin=dict(l=65, r=30, t=60, b=110), plot_bgcolor='#fafafa',
                  bargap=0.25, bargroupgap=0.08)
save(fig, "02_conditional_bars", h=620)

# ===================================================================
#  3.  DATA OVERVIEW — Correlation Heatmap
# ===================================================================
cols = LAG1_FEATURE_COLS
names = [friendly_name(c).replace(' (prev day)', '') for c in cols]
nc = len(cols)
agreement = np.zeros((nc, nc))
for i in range(nc):
    for j in range(nc):
        agreement[i, j] = (df_model[cols[i]] == df_model[cols[j]]).mean() * 100
fig = go.Figure(data=go.Heatmap(z=agreement, x=names, y=names, colorscale='Blues',
                                text=[[f'{v:.0f}%' for v in row] for row in agreement],
                                texttemplate='%{text}', textfont=dict(size=11),
                                colorbar=dict(title='Agreement %')))
fig.update_layout(height=560, margin=dict(l=130, r=30, t=20, b=110), xaxis_tickangle=-35)
save(fig, "03_correlation_heatmap", h=560)

# ===================================================================
#  4.  DATA OVERVIEW — Global Markets Time Series (default 3 indices)
# ===================================================================
default_indices = [c for c in ['NIFTY_CHG_PCT_1D', 'DJ_CHG_PCT_1D', 'SP_CHG_PCT_1D']
                   if c in df_raw_pct.columns]
fig = go.Figure()
for col in default_indices:
    prefix = col.split('_CHG_PCT')[0]
    nm = INDEX_NAMES.get(prefix, prefix)
    if col in df_raw_pct.columns:
        fig.add_trace(go.Scatter(x=df_raw_pct['Dates'], y=df_raw_pct[col],
                                 mode='lines', name=nm, opacity=0.8))
fig.update_layout(height=450, yaxis_title='Daily % Change', xaxis_title='Date',
                  legend=dict(orientation='h', y=-0.15), margin=dict(l=50, r=20, t=20, b=80),
                  hovermode='x unified')
save(fig, "04_market_timeseries", h=450)

# ===================================================================
#  DECISION TREE  (depth=3, test=20%)
# ===================================================================
print("\n─── Decision Tree ───")
X_train, X_test, y_train, y_test = get_Xy(DT_TEST)
dt_clf = DecisionTreeClassifier(max_depth=DT_DEPTH, random_state=42)
dt_clf.fit(X_train, y_train)
dt_pred = dt_clf.predict(X_test)
friendly_names = [friendly_name(f) for f in feat_names]

# 5. Tree Visualisation
tree = dt_clf.tree_
tree_depth = dt_clf.get_depth()
Y_STEP = 3.0; BOX_W = 1.6; BOX_H = 1.1; FONT_SIZE = 11
dx_init = max(2.0, (2 ** tree_depth) * BOX_W * 0.65)

def get_tree_data(node, x, y, dx, depth=0):
    nodes_data, edges_data = [], []
    n_samples = tree.n_node_samples[node]
    values = tree.value[node][0]
    majority = 'UP' if values[1] >= values[0] else 'DOWN'
    color = '#22c55e' if values[1] >= values[0] else '#ef4444'
    gini = tree.impurity[node]
    if tree.children_left[node] == -1:
        label = (f'<b>{majority}</b><br>Gini={gini:.3f}<br>'
                 f'{int(values[0])}D / {int(values[1])}U<br>n={n_samples}')
    else:
        fi = tree.feature[node]; th = tree.threshold[node]
        fn = friendly_names[fi] if fi < len(friendly_names) else f'Feature {fi}'
        label = (f'<b>{fn}</b><br>≤ {th:.1f} ?<br>Gini={gini:.3f}<br>'
                 f'{int(values[0])}D / {int(values[1])}U<br>n={n_samples}')
    nodes_data.append((x, y, label, color, tree.children_left[node] == -1))
    if tree.children_left[node] != -1:
        left, right = tree.children_left[node], tree.children_right[node]
        new_dx = dx / 2
        lx, ly = x - dx, y - Y_STEP
        edges_data.append((x, y, lx, ly, 'Yes'))
        ln, le = get_tree_data(left, lx, ly, new_dx, depth + 1)
        nodes_data.extend(ln); edges_data.extend(le)
        rx, ry = x + dx, y - Y_STEP
        edges_data.append((x, y, rx, ry, 'No'))
        rn, re = get_tree_data(right, rx, ry, new_dx, depth + 1)
        nodes_data.extend(rn); edges_data.extend(re)
    return nodes_data, edges_data

nodes, edges = get_tree_data(0, 0, 0, dx_init)
fig = go.Figure()
all_x = [n[0] for n in nodes]; all_y = [n[1] for n in nodes]
annotations = []
for x1, y1, x2, y2, lbl in edges:
    fig.add_trace(go.Scatter(x=[x1, x2], y=[y1 - BOX_H, y2 + BOX_H], mode='lines',
                             line=dict(color='#94a3b8', width=1.5), showlegend=False, hoverinfo='skip'))
    annotations.append(dict(x=(x1+x2)/2, y=(y1-BOX_H+y2+BOX_H)/2, text=f'<b>{lbl}</b>',
                            showarrow=False, font=dict(size=11, color='#475569'), bgcolor='white', borderpad=2))
for x, y, label, color, is_leaf in nodes:
    fill = color if is_leaf else '#f8fafc'
    tc = '#ffffff' if is_leaf else '#1e293b'
    annotations.append(dict(x=x, y=y, xref='x', yref='y', text=label, showarrow=False,
                            font=dict(size=FONT_SIZE, color=tc, family='Calibri, Arial, sans-serif'),
                            align='center', bgcolor=fill, bordercolor=color, borderwidth=2, borderpad=8))
x_pad = BOX_W * 2.5; y_pad = BOX_H * 2.5
fig.update_layout(annotations=annotations, height=max(520, (tree_depth+1)*180),
                  xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                             range=[min(all_x)-x_pad, max(all_x)+x_pad]),
                  yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                             range=[min(all_y)-y_pad, max(all_y)+y_pad]),
                  margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor='white')
save(fig, "05_dt_tree_visualisation", h=max(520, (tree_depth+1)*180))

# 6. DT Feature Space (default: first two features)
fx, fy = LAG1_FEATURE_COLS[0], LAG1_FEATURE_COLS[1]
fi_x, fi_y = feat_names.index(fx), feat_names.index(fy)
X_full = np.concatenate([X_train, X_test]); y_full = np.concatenate([y_train, y_test])
fx_name = friendly_name(fx).replace(' (prev day)', ''); fy_name = friendly_name(fy).replace(' (prev day)', '')
subplot_map = {(0,1):(1,1),(1,1):(1,2),(0,0):(2,1),(1,0):(2,2)}
h_sp=0.20; v_sp=0.28; cw=(1.0-h_sp)/2.0; ch=(1.0-v_sp)/2.0
fig = go.Figure(); pad_inner=0.03
domains_raw = {(1,1):(0.0,cw,1.0-ch,1.0),(1,2):(cw+h_sp,1.0,1.0-ch,1.0),
               (2,1):(0.0,cw,0.0,ch),(2,2):(cw+h_sp,1.0,0.0,ch)}
domains_pie = {k:(x0+pad_inner,x1-pad_inner,y0+pad_inner,y1-pad_inner) for k,(x0,x1,y0,y1) in domains_raw.items()}
anns=[]; shps=[]
for (vx,vy) in [(0,1),(1,1),(0,0),(1,0)]:
    mask = (X_full[:,fi_x]==vx)&(X_full[:,fi_y]==vy)
    n_total=int(mask.sum()); y_sub=y_full[mask]; n_up=int(y_sub.sum()); n_down=n_total-n_up
    sample=np.zeros((1,len(feat_names))); sample[0,fi_x]=vx; sample[0,fi_y]=vy
    tree_pred=int(dt_clf.predict(sample)[0]); pred_word='UP' if tree_pred==1 else 'DOWN'
    if n_total>0:
        verdict_text='CORRECT' if (tree_pred==1 and n_up>=n_down) or (tree_pred==0 and n_down>=n_up) else 'MISMATCH'
        verdict_color='#16a34a' if verdict_text=='CORRECT' else '#f59e0b'
    else: verdict_text='—'; verdict_color='#64748b'
    row,col=subplot_map[(vx,vy)]; px0,px1,py0,py1=domains_pie[(row,col)]
    fig.add_trace(go.Pie(values=[n_up,n_down] if n_total>0 else [1],
                         labels=['NIFTY UP','NIFTY DOWN'] if n_total>0 else ['No data'],
                         marker=dict(colors=['#22c55e','#ef4444'] if n_total>0 else ['#e5e7eb'],
                                     line=dict(color='white',width=2)),
                         hole=0.50, textinfo='label+value', textposition='inside',
                         textfont=dict(size=9), showlegend=False,
                         domain=dict(x=[px0,px1],y=[py0,py1])))
    bx0,bx1,by0,by1=domains_raw[(row,col)]
    anns.append(dict(x=(bx0+bx1)/2,y=by0-0.03,xref='paper',yref='paper',
                     text=f'<span style="font-size:11px;color:#64748b">{n_total} days</span>  '
                          f'<b style="font-size:11px;color:{verdict_color}">{verdict_text}</b>', showarrow=False))
    bc='#22c55e' if tree_pred==1 else '#ef4444'
    fc='rgba(34,197,94,0.04)' if tree_pred==1 else 'rgba(239,68,68,0.04)'
    shps.append(dict(type='rect',xref='paper',yref='paper',x0=bx0,x1=bx1,y0=by0,y1=by1,
                     line=dict(color=bc,width=2),fillcolor=fc,layer='below'))
left_cx=cw/2; right_cx=cw+h_sp+cw/2; top_y=1.0+0.06
anns.append(dict(x=left_cx,y=top_y,xref='paper',yref='paper',
                 text=f'<b style="color:#dc2626">{fx_name} DOWN</b>',showarrow=False,font=dict(size=13)))
anns.append(dict(x=right_cx,y=top_y,xref='paper',yref='paper',
                 text=f'<b style="color:#16a34a">{fx_name} UP</b>',showarrow=False,font=dict(size=13)))
anns.append(dict(x=-0.07,y=1.0-ch/2,xref='paper',yref='paper',
                 text=f'<b style="color:#16a34a">{fy_name}<br>UP</b>',showarrow=False,font=dict(size=12),textangle=-90))
anns.append(dict(x=-0.07,y=ch/2,xref='paper',yref='paper',
                 text=f'<b style="color:#dc2626">{fy_name}<br>DOWN</b>',showarrow=False,font=dict(size=12),textangle=-90))
fig.update_layout(height=720,margin=dict(l=90,r=20,t=60,b=50),plot_bgcolor='white',annotations=anns,shapes=shps)
save(fig, "06_dt_feature_space", h=720)

# 7. DT Confusion Matrix
save(make_confusion_fig(y_test, dt_pred, 'Decision Tree'), "07_dt_confusion", h=350)

# 8. DT ROC Curve
dt_prob = dt_clf.predict_proba(X_test)[:, 1]
save(make_roc_fig(y_test, dt_prob, 'Decision Tree'), "08_dt_roc", h=480)

# 9. DT Error vs Tree Size
max_sizes = list(range(1, 19)); n_cv = 5
train_errors=[]; cv_errors=[]; cv_stds=[]; test_errors=[]; train_stds=[]; test_stds_l=[]
for d in max_sizes:
    c = DecisionTreeClassifier(max_depth=d, random_state=42); c.fit(X_train, y_train)
    train_errors.append(1.0 - accuracy_score(y_train, c.predict(X_train)))
    test_errors.append(1.0 - accuracy_score(y_test, c.predict(X_test)))
    cv = cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=42), X_train, y_train, cv=n_cv, scoring='accuracy')
    cv_errors.append(1.0 - cv.mean()); cv_stds.append(cv.std())
    train_stds.append(cv.std()*0.4); test_stds_l.append(cv.std()*0.8)
fig = go.Figure()
fig.add_trace(go.Scatter(x=max_sizes, y=train_errors, mode='lines+markers', name='Training',
              line=dict(color='#1a1a1a',width=2.5), marker=dict(size=7,symbol='circle-open'),
              error_y=dict(type='data',array=train_stds,visible=True,color='rgba(26,26,26,0.35)')))
fig.add_trace(go.Scatter(x=max_sizes, y=cv_errors, mode='lines+markers', name='Cross-Validation',
              line=dict(color='#e05500',width=2.5,dash='dash'), marker=dict(size=7,symbol='circle-open'),
              error_y=dict(type='data',array=cv_stds,visible=True,color='rgba(224,85,0,0.35)')))
fig.add_trace(go.Scatter(x=max_sizes, y=test_errors, mode='lines+markers', name='Test',
              line=dict(color='#009688',width=2.5,dash='dot'), marker=dict(size=7,symbol='circle-open'),
              error_y=dict(type='data',array=test_stds_l,visible=True,color='rgba(0,150,136,0.35)')))
idx=max_sizes.index(DT_DEPTH)
fig.add_vline(x=DT_DEPTH, line_dash='dash', line_color='rgba(100,100,100,0.4)', line_width=1)
fig.add_annotation(x=DT_DEPTH, y=max(train_errors[idx],cv_errors[idx],test_errors[idx])+0.03,
                   text=f'<b>Selected depth = {DT_DEPTH}</b>', showarrow=True, arrowhead=2,
                   font=dict(size=11), bgcolor='rgba(255,255,255,0.9)', bordercolor='#999', borderwidth=1)
fig.update_layout(xaxis_title='Tree Size (max depth)', yaxis_title='Error (1 − Accuracy)',
                  height=480, margin=dict(l=60,r=30,t=30,b=60),
                  legend=dict(orientation='h',y=1.08,x=0.5,xanchor='center'),
                  plot_bgcolor='#fafafa', xaxis=dict(showgrid=True,gridcolor='#e5e7eb',dtick=1),
                  yaxis=dict(showgrid=True,gridcolor='#e5e7eb',rangemode='tozero'))
save(fig, "09_dt_error_vs_size", h=480)

# ===================================================================
#  RANDOM FOREST  (100 trees, depth=4, test=20%)
# ===================================================================
print("\n─── Random Forest ───")
X_train, X_test, y_train, y_test = get_Xy(RF_TEST)
rf_clf = RandomForestClassifier(n_estimators=RF_TREES, max_depth=RF_DEPTH, random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)

# 10. RF Feature Importance
imp = rf_clf.feature_importances_; mx = imp.max()
scaled = (imp/mx*100) if mx>0 else imp; idxs = np.argsort(scaled)
fig = go.Figure()
fig.add_trace(go.Bar(y=[friendly_name(feat_names[i]) for i in idxs], x=scaled[idxs], orientation='h',
                     marker_color='#dc2626', text=[f'{v:.0f}' for v in scaled[idxs]], textposition='outside'))
fig.update_layout(height=500, xaxis_title='Variable Importance', xaxis_range=[0,110],
                  margin=dict(l=200,r=40,t=20,b=50))
save(fig, "10_rf_importance", h=500)

# 11. RF Confusion
save(make_confusion_fig(y_test, rf_pred, 'Random Forest'), "11_rf_confusion", h=350)

# 12. RF ROC
rf_prob = rf_clf.predict_proba(X_test)[:, 1]
save(make_roc_fig(y_test, rf_prob, 'Random Forest'), "12_rf_roc", h=480)

# 13. RF Learning Curve (Trees vs Error)
tree_counts = list(range(10, RF_TREES+1, 10)); te_list=[]; oob_list=[]
for nt in tree_counts:
    rf = RandomForestClassifier(n_estimators=nt, max_depth=RF_DEPTH, random_state=42, n_jobs=-1, oob_score=True)
    rf.fit(X_train, y_train)
    te_list.append(1-accuracy_score(y_test, rf.predict(X_test))); oob_list.append(1-rf.oob_score_)
fig = go.Figure()
fig.add_trace(go.Scatter(x=tree_counts, y=te_list, mode='lines', name='Test Error', line=dict(color='#1e293b',width=2)))
fig.add_trace(go.Scatter(x=tree_counts, y=oob_list, mode='lines', name='OOB Error', line=dict(color='#14b8a6',width=2)))
fig.add_hline(y=0.5, line_dash='dash', line_color='gray', annotation_text='50% (coin flip)')
fig.update_layout(height=450, xaxis_title='Number of Trees', yaxis_title='Error',
                  yaxis_range=[0, max(max(te_list),max(oob_list))*1.15],
                  legend=dict(orientation='h',y=-0.15), margin=dict(l=50,r=20,t=20,b=60))
save(fig, "13_rf_trees_vs_error", h=450)

# 14. RF Max Features Chart
p = X_train.shape[1]
m_settings = {f'All {p} clues (Bagging)': (p,'#f59e0b'),
              f'Half ({p//2} clues)': (p//2,'#3b82f6'),
              f'Square root ({int(math.sqrt(p))} clues — RF default)': (int(math.sqrt(p)),'#14b8a6')}
fig = go.Figure()
for label,(m_val,clr) in m_settings.items():
    errs=[]
    for nt in tree_counts:
        rf = RandomForestClassifier(n_estimators=nt,max_depth=RF_DEPTH,max_features=m_val,random_state=42,n_jobs=-1)
        rf.fit(X_train, y_train); errs.append(1-accuracy_score(y_test, rf.predict(X_test)))
    fig.add_trace(go.Scatter(x=tree_counts,y=errs,mode='lines',name=label,line=dict(color=clr,width=2)))
fig.update_layout(height=450, xaxis_title='Number of Trees', yaxis_title='Test Classification Error',
                  legend=dict(orientation='h',y=-0.15), margin=dict(l=50,r=20,t=20,b=60))
save(fig, "14_rf_max_features", h=450)

# ===================================================================
#  GRADIENT BOOSTING  (100 trees, depth=3, lr=0.1, test=20%)
# ===================================================================
print("\n─── Gradient Boosting ───")
X_train, X_test, y_train, y_test = get_Xy(GB_TEST)
gb_clf = GradientBoostingClassifier(n_estimators=GB_TREES, max_depth=GB_DEPTH, learning_rate=GB_LR, random_state=42)
gb_clf.fit(X_train, y_train)
gb_pred = gb_clf.predict(X_test)

# 15. GB Feature Importance
imp = gb_clf.feature_importances_; mx = imp.max()
scaled = (imp/mx*100) if mx>0 else imp; idxs = np.argsort(scaled)
fig = go.Figure()
fig.add_trace(go.Bar(y=[friendly_name(feat_names[i]) for i in idxs], x=scaled[idxs], orientation='h',
                     marker_color='#f59e0b', text=[f'{v:.0f}' for v in scaled[idxs]], textposition='outside'))
fig.update_layout(height=500, xaxis_title='Variable Importance', xaxis_range=[0,110],
                  margin=dict(l=200,r=40,t=20,b=50))
save(fig, "15_gb_importance", h=500)

# 16. GB Confusion
save(make_confusion_fig(y_test, gb_pred, 'Gradient Boosting'), "16_gb_confusion", h=350)

# 17. GB ROC
gb_prob = gb_clf.predict_proba(X_test)[:, 1]
save(make_roc_fig(y_test, gb_prob, 'Gradient Boosting'), "17_gb_roc", h=480)

# 18. GB Staged Accuracy
stages = list(range(1, gb_clf.n_estimators+1))
tr_sc=[accuracy_score(y_train,p) for p in gb_clf.staged_predict(X_train)]
te_sc=[accuracy_score(y_test,p) for p in gb_clf.staged_predict(X_test)]
fig = go.Figure()
fig.add_trace(go.Scatter(x=stages,y=tr_sc,mode='lines',name='Train',line=dict(color='#f59e0b')))
fig.add_trace(go.Scatter(x=stages,y=te_sc,mode='lines',name='Test',line=dict(color='#ef4444')))
fig.add_hline(y=0.5, line_dash='dash', line_color='gray', annotation_text='50% baseline')
fig.update_layout(height=400, xaxis_title='Number of Boosting Stages', yaxis_title='Accuracy',
                  legend=dict(orientation='h',y=-0.15), margin=dict(l=50,r=20,t=20,b=60))
save(fig, "18_gb_staged_accuracy", h=400)

# 19. GB Boosting vs Error
test_errs = [1-accuracy_score(y_test,p) for p in gb_clf.staged_predict(X_test)]
stump = DecisionTreeClassifier(max_depth=1, random_state=42); stump.fit(X_train, y_train)
stump_err = 1-accuracy_score(y_test, stump.predict(X_test))
full_tree = DecisionTreeClassifier(max_depth=None, random_state=42); full_tree.fit(X_train, y_train)
full_err = 1-accuracy_score(y_test, full_tree.predict(X_test)); full_nodes = full_tree.tree_.node_count
fig = go.Figure()
fig.add_trace(go.Scatter(x=stages,y=test_errs,mode='lines',name='Boosting',line=dict(color='#ea580c',width=2.5)))
fig.add_trace(go.Scatter(x=[stages[0],stages[-1]],y=[stump_err,stump_err],mode='lines',
              name='Single Stump (depth 1)',line=dict(color='#1e293b',width=1.5,dash='dot')))
fig.add_annotation(x=stages[-1]*0.65,y=stump_err+0.015,text='Single Stump',showarrow=False,font=dict(size=12))
fig.add_trace(go.Scatter(x=[stages[0],stages[-1]],y=[full_err,full_err],mode='lines',
              name=f'{full_nodes} Node Tree',line=dict(color='#1e293b',width=1.5,dash='dot')))
fig.add_annotation(x=stages[-1]*0.65,y=full_err+0.015,text=f'{full_nodes} Node Tree',showarrow=False,font=dict(size=12))
fig.update_layout(height=450, xaxis_title='Boosting Iterations', yaxis_title='Test Error',
                  yaxis_range=[0, max(stump_err,full_err,max(test_errs))*1.15],
                  legend=dict(orientation='h',y=-0.18), margin=dict(l=60,r=30,t=20,b=70),
                  plot_bgcolor='white', xaxis=dict(showgrid=True,gridcolor='#e5e7eb'),
                  yaxis=dict(showgrid=True,gridcolor='#e5e7eb'))
save(fig, "19_gb_boosting_vs_error", h=450)

# ===================================================================
#  COMPARE MODELS
# ===================================================================
print("\n─── Compare Models ───")
X_train, X_test, y_train, y_test = get_Xy(CMP_TEST)
X_all = df_model[LAG1_FEATURE_COLS].values; y_all = df_model['NIFTY_Direction'].values
models_cmp = {
    'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42),
}
results = {}
for name, mdl in models_cmp.items():
    mdl.fit(X_train, y_train); yp = mdl.predict(X_test)
    cv = cross_val_score(mdl, X_all, y_all, cv=5, scoring='accuracy')
    results[name] = {'accuracy': accuracy_score(y_test,yp), 'precision': precision_score(y_test,yp,zero_division=0),
                     'recall': recall_score(y_test,yp,zero_division=0), 'f1': f1_score(y_test,yp,zero_division=0),
                     'cv_mean': cv.mean(), 'cv_std': cv.std()}

# 20. Compare Models Bar Chart
model_names = list(results.keys())
metrics = ['accuracy','precision','recall','f1']
colors = {'accuracy':'#3b82f6','precision':'#22c55e','recall':'#f59e0b','f1':'#8b5cf6'}
fig = go.Figure()
for m in metrics:
    vals = [results[mn][m] for mn in model_names]
    fig.add_trace(go.Bar(name=m.capitalize(), x=model_names, y=vals, marker_color=colors[m],
                         text=[f'{v:.1%}' for v in vals], textposition='outside'))
fig.add_hline(y=0.5, line_dash='dash', line_color='red', annotation_text='50% baseline')
fig.update_layout(barmode='group', height=450, yaxis_title='Score', yaxis_range=[0,1],
                  legend=dict(orientation='h',y=-0.15), margin=dict(l=50,r=20,t=20,b=60))
save(fig, "20_compare_models", h=450)

print(f"\n✅ All 20 plots saved to: {OUT}")
