import plotly.graph_objects as go

# 测试数据
sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
# 将执行时间乘以1000
time_v0 = [t * 1000 for t in [0.002886, 0.00319447, 0.00345337, 0.00437669, 0.00618377, 0.00974218, 0.0170282]]
time_v1 = [t * 1000 for t in [0.00249573, 0.00281514, 0.00354412, 0.0049911, 0.00794745, 0.0137924, 0.0254141]]
time_v2 = [t * 1000 for t in [0.00250465, 0.00282348, 0.00355178, 0.00510852, 0.00808564, 0.0140345, 0.0259723]]
time_v3 = [t * 1000 for t in [0.002481, 0.00244345, 0.00246507, 0.00295557, 0.00386007, 0.00571989, 0.00932561]]

# 计算加速比
speedup_v1 = [t0/t1 for t0, t1 in zip(time_v0, time_v1)]
speedup_v2 = [t0/t2 for t0, t2 in zip(time_v0, time_v2)]
speedup_v3 = [t0/t3 for t0, t3 in zip(time_v0, time_v3)]

# 将sizes分成两组
sizes_row1 = sizes[:3]  # [128, 256, 512]
sizes_row2 = sizes[3:]  # [1024, 2048, 4096, 8192]

# 创建执行时间图
fig1 = go.Figure()

# 添加第一行的执行时间柱状图
bar_width = 0.15
for i, (data, name, color) in enumerate(zip(
    [time_v0, time_v1, time_v2, time_v3],
    ['V0 (基础版本)', 'V1 (Warp优化)', 'V2 (CUB WarpReduce)', 'V3 (Float4 + BlockReduce)'],
    ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)']
)):
    # 第一行数据
    fig1.add_trace(go.Bar(
        name=name,
        x=[str(s) for s in sizes_row1],
        y=data[:3],
        width=bar_width,
        offset=-bar_width*1.5 + i*bar_width,
        marker_color=color,
        text=[f'{x:.2f}' for x in data[:3]],  # 修改这里
        textposition='outside',
        textangle=0,
        xaxis='x',
        yaxis='y'
    ))
    
    # 第二行数据
    fig1.add_trace(go.Bar(
        name=name,
        x=[str(s) for s in sizes_row2],
        y=data[3:],
        width=bar_width,
        offset=-bar_width*1.5 + i*bar_width,
        marker_color=color,
        text=[f'{x:.2f}' for x in data[3:]],  # 修改这里
        textposition='outside',
        textangle=0,
        xaxis='x2',
        yaxis='y2',
        showlegend=False
    ))

# 更新第一张图的布局
fig1.update_layout(
    width=1500,
    height=800,
    title_text="RMSNORM CUDA优化版本性能比较 (1000次执行)",
    title_x=0.5,
    template='plotly_white',
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=1.1,
        xanchor="right",
        x=0.99
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1,
    
    # 设置两个子图的网格
    grid=dict(
        rows=2, 
        columns=1,
        pattern='independent',
        roworder='top to bottom'
    ),
    
    # 配置两个x轴和y轴
    xaxis=dict(
        title="向量大小",
        type='category',
        domain=[0, 1],
        anchor='y'
    ),
    xaxis2=dict(
        title="向量大小",
        type='category',
        domain=[0, 1],
        anchor='y2'
    ),
    yaxis=dict(
        title="1000次执行时间 (ms)",
        type="log",
        domain=[0.55, 1]
    ),
    yaxis2=dict(
        title="1000次执行时间 (ms)",
        type="log",
        domain=[0, 0.45]
    )
)

# 对第二张图做类似的修改
fig2 = go.Figure()

for i, (data, name, color) in enumerate(zip(
    [speedup_v1, speedup_v2, speedup_v3],
    ['V1 加速比', 'V2 加速比', 'V3 加速比'],
    ['rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)']
)):
    # 第一行数据
    fig2.add_trace(go.Bar(
        name=name,
        x=[str(s) for s in sizes_row1],
        y=data[:3],
        width=bar_width,
        offset=-bar_width + i*bar_width,
        marker_color=color,
        text=[f'{x:.2f}x' for x in data[:3]],
        textposition='outside',
        textangle=0,
        xaxis='x',
        yaxis='y'
    ))
    
    # 第二行数据
    fig2.add_trace(go.Bar(
        name=name,
        x=[str(s) for s in sizes_row2],
        y=data[3:],
        width=bar_width,
        offset=-bar_width + i*bar_width,
        marker_color=color,
        text=[f'{x:.2f}x' for x in data[3:]],
        textposition='outside',
        textangle=0,
        xaxis='x2',
        yaxis='y2',
        showlegend=False
    ))

# 添加基准线
for xaxis, yaxis in [('x', 'y'), ('x2', 'y2')]:
    fig2.add_trace(go.Scatter(
        x=sizes_row1 if xaxis == 'x' else sizes_row2,
        y=[1]*len(sizes_row1 if xaxis == 'x' else sizes_row2),
        name='基准线',
        line=dict(dash='dash', color='red'),
        xaxis=xaxis,
        yaxis=yaxis,
        showlegend=(xaxis == 'x')
    ))

# 更新第二张图的布局
fig2.update_layout(
    width=1500,
    height=800,
    title_text="各版本相对于基础版本的加速比",
    title_x=0.5,
    template='plotly_white',
    showlegend=True,
    legend=dict(
        yanchor="top",
        y=1.1,
        xanchor="right",
        x=0.99
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1,
    
    # 设置两个子图的网格
    grid=dict(
        rows=2, 
        columns=1,
        pattern='independent',
        roworder='top to bottom'
    ),
    
    # 配置两个x轴和y轴
    xaxis=dict(
        title="向量大小",
        type='category',
        domain=[0, 1],
        anchor='y'
    ),
    xaxis2=dict(
        title="向量大小",
        type='category',
        domain=[0, 1],
        anchor='y2'
    ),
    yaxis=dict(
        title="相对V0的加速比",
        domain=[0.55, 1]
    ),
    yaxis2=dict(
        title="相对V0的加速比",
        domain=[0, 0.45]
    )
)

# 更新两张图的x轴
for fig in [fig1, fig2]:
    fig.update_xaxes(
        type='category',
        ticktext=sizes,
        tickvals=sizes,
    )

# 保存图表
fig1.write_image('rmsnorm_performance_time.png', width=1500, height=800)
fig1.write_html('rmsnorm_performance_time.html')

fig2.write_image('rmsnorm_performance_speedup.png', width=1500, height=800)
fig2.write_html('rmsnorm_performance_speedup.html')