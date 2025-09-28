import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

DATA_FILENAME = Path(__file__).parent / 'data/public_dashboard_data.jsonl'

DATA_FIELD_RENAMES = {
    'platform': 'Platform',
    'conspiracy_classification': 'Conspiracy Classification',
    'cluster_id': 'Cluster ID',
    'cluster_theme': 'Theme',
    'date_posted': 'Date Posted',
    'mapping_x': 'X',
    'mapping_y': 'Y'
}

st.set_page_config(
    page_title='Conspiracy Mapping Dashboard',
    page_icon=':frame_with_picture:',
    layout='wide',
)

@st.cache_data
def get_data() -> pd.DataFrame:
    raw_df = pd.read_json(DATA_FILENAME, lines=True)
    raw_df.rename(columns=DATA_FIELD_RENAMES, inplace=True)

    if 'Date Posted' in raw_df.columns: 
        raw_df['Date Posted'] = pd.to_datetime(raw_df['Date Posted'], utc=True)

    return raw_df

def make_period_series(dates: pd.Series, freq: str) -> pd.Series:
    if freq == 'Daily':
        return dates.dt.to_period('D').astype(str)
    if freq == 'Weekly':
        return dates.dt.to_period('W').astype(str)
    if freq == 'Monthly':
        return dates.dt.to_period('M').astype(str)
    if freq == 'Quarterly':
        return dates.dt.to_period('Q').astype(str)
    if freq == 'Yearly':
        return dates.dt.to_period('Y').astype(str)
    raise ValueError(f"Unsupported frequency: {freq}")


def sidebar_controls(df: pd.DataFrame) -> dict:
    st.sidebar.header('Controls')
    freq = st.sidebar.selectbox('Time bin for animation', options=['Daily', 'Weekly', 'Monthly'], index=2)

    window_size_input = st.sidebar.number_input('Window size (periods)', min_value=1, value=2, step=1)
    stride_input = st.sidebar.number_input('Stride (periods)', min_value=1, value=1, step=1, help='Number of periods to move the window forward for each frame, i.e., how much the window overlaps with the previous frame.')
    visualise_options = ['Platform', 'Conspiracy Classification', 'Cluster ID', 'Theme']
    visualise = st.sidebar.selectbox('Visualise category', options=visualise_options, index=3)
    point_size = st.sidebar.slider('Point size (px)', min_value=1, max_value=40, value=2)
    opacity = st.sidebar.slider('Point opacity', min_value=0.05, max_value=1.0, value=0.6, step=0.05)
    animation_speed = st.sidebar.slider('Animation frame duration (ms)', min_value=100, max_value=5000, value=1000, step=50)
    
    st.sidebar.markdown('---')
    platforms = st.sidebar.multiselect('Platform', options=sorted(df['Platform'].dropna().unique()), default=sorted(df['Platform'].dropna().unique()))
    classifs = st.sidebar.multiselect('Conspiracy Classification', options=sorted(df['Conspiracy Classification'].dropna().unique()), default=sorted(df['Conspiracy Classification'].dropna().unique()))
    clusters = st.sidebar.multiselect('Cluster ID', options=sorted(df['Cluster ID'].dropna().unique()), default=sorted(df['Cluster ID'].dropna().unique()))
    themes = st.sidebar.multiselect('Theme', options=sorted(df['Theme'].dropna().unique()), default=sorted(df['Theme'].dropna().unique()))

    st.sidebar.markdown('---')
    
    return dict(
        freq=freq,
        window_size_input=window_size_input,
        stride_input=stride_input,
        visualise=visualise,
        point_size=point_size,
        opacity=opacity,
        animation_speed=animation_speed,
        platforms=platforms,
        classifs=classifs,
        clusters=clusters,
        themes=themes
    )


def filter_data(df: pd.DataFrame, controls: dict) -> tuple[pd.DataFrame, list[str]]:
    """
    Filter the DataFrame based on the controls and return the filtered DataFrame and the ordered list of periods.
    
    The list of periods is ordered chronologically based on the minimum date_posted in each period.    
    """
    mask = (
        (df['Platform'].isin(controls['platforms'])) &
        (df['Conspiracy Classification'].isin(controls['classifs'])) &
        (df['Cluster ID'].isin(controls['clusters'])) &
        (df['Theme'].isin(controls['themes']))
    )
    filtered = df.loc[mask].copy()
    if filtered.empty:
        st.warning('No records match the selected filters and date range.')
        st.stop()
    filtered['_period'] = make_period_series(filtered['Date Posted'], controls['freq'])
    period_order = (
        filtered.groupby('_period')['Date Posted']
        .min()
        .sort_values()
        .index
        .tolist()
    )
    if period_order:
        filtered['_period'] = pd.Categorical(filtered['_period'], categories=period_order, ordered=True)
    return filtered, period_order


def build_animation_df(filtered: pd.DataFrame, period_order: list[str], window_size_input: int, stride_input: int) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a DataFrame suitable for animation, with a '_frame_label' column indicating the frame each record belongs to.
    
    The frame labels are of the form "start_period → end_period".
    """
    period_count = len(period_order)
    window_size = int(max(1, min(window_size_input, period_count)))
    stride = int(max(1, min(stride_input, period_count)))
    frame_starts = list(range(0, max(1, period_count - window_size + 1), stride))
    if period_count > 0 and not frame_starts:
        frame_starts = [0]
    frame_labels = []
    frames = []
    for s in frame_starts:
        e = min(s + window_size - 1, period_count - 1)
        label = f"{period_order[s]} → {period_order[e]}"
        frame_labels.append(label)
        period_set = set(period_order[s : e + 1])
        frame_mask = filtered['_period'].isin(period_set)
        if frame_mask.any():
            tmp = filtered.loc[frame_mask].copy()
            tmp['_frame_label'] = label
            frames.append(tmp)
    if frames:
        anim_df = pd.concat(frames, ignore_index=True)
    else:
        anim_df = filtered.copy()
        anim_df['_frame_label'] = f"{period_order[0]} → {period_order[-1]}" if period_order else 'all'
    return anim_df, frame_labels


def make_traces_for_label(anim_df, lbl: str, color_by: str, color_map: dict, dist_cat: str, dist_cats: list[str], dist_color_map: dict, point_size: int, opacity: float):
    """
    Create scatter and bar chart traces for a given frame label. 
    
    Arguments:
    - anim_df: DataFrame containing the animation data with a '_frame_label' column.
    - lbl: The frame label to filter the DataFrame. (e.g., "2021-01 → 2021-03")
    - color_by: The column name to color the scatter plot points by.
    - color_map: A dictionary mapping category values to colors for the scatter plot.
    - dist_cat: The column name to use for the distribution bar chart
    - dist_cats: A list of all possible categories for the distribution bar chart (should match keys in dist_color_map).
    - dist_color_map: A dictionary mapping category values to colors for the bar chart (should match dist_cats).
    - point_size: Size of the scatter plot points in pixels (integer).
    - opacity: Opacity of the scatter plot points (0.0 to 1.0).
    
    Returns: scatter_trace, bar_trace
    """
    sdf = anim_df[anim_df['_frame_label'] == lbl]
    if sdf.empty:
        scatter = go.Scatter(x=[], y=[], mode='markers', marker=dict(size=point_size, opacity=opacity), showlegend=False)
        bar = go.Bar(x=[], y=[], orientation='h', showlegend=False)
        return scatter, bar

    # safe color lookup: color_map may be None if no categories were found
    marker_colors = [color_map.get(v, '#888') if color_map else '#888' for v in sdf[color_by]]
    
    # Scatter plot
    scatter = go.Scatter(
        x=sdf['X'],
        y=sdf['Y'],
        mode='markers',
        marker=dict(size=point_size, opacity=opacity, color=marker_colors),
        hovertext=sdf[['Platform', 'Conspiracy Classification', 'Cluster ID', 'Theme', 'Date Posted']].astype(str).agg(' | '.join, axis=1),
        showlegend=False,
    )
        
    # Bar chart: count values for the requested distribution column
    ddf = sdf[dist_cat].fillna('Unknown') if dist_cat in sdf.columns else pd.Series([], dtype=object)
    vc = ddf.value_counts()
    counts_map = {c: int(vc.get(c, 0)) for c in dist_cats}
    cats_order = sorted(dist_cats, key=lambda c: counts_map[c], reverse=True)
    values = [counts_map[c] for c in cats_order]
    colors = [dist_color_map.get(c, '#888') for c in cats_order]
    bar = go.Bar(x=values, y=cats_order, orientation='h', marker_color=colors, showlegend=False)

    return scatter, bar


def plot_dashboard(anim_df: pd.DataFrame, frame_labels: list[str], filtered: pd.DataFrame, controls: dict) -> None:
    """
    Plot the dashboard with animation controls.
    
    Arguments:
    - anim_df: DataFrame containing the animation data with a '_frame_label' column.
    - frame_labels: List of frame labels in the order they should appear in the animation.
    - filtered: The filtered DataFrame used to determine axis ranges.
    - controls: Dictionary of control settings from the sidebar.
    """
    color_seq = px.colors.qualitative.Plotly
    color_by = controls['visualise']
    dist_cat = controls['visualise']
    point_size = controls['point_size']
    opacity = controls['opacity']
    animation_speed = controls['animation_speed']
    color_map = None
    
    if color_by in anim_df.columns:
        cats = sorted(anim_df[color_by].dropna().unique())
        color_map = {c: color_seq[i % len(color_seq)] for i, c in enumerate(cats)}
    
    dist_color_map = {}
    
    if dist_cat in anim_df.columns:
        dist_cats = sorted(anim_df[dist_cat].fillna('Unknown').unique())
        dist_color_map = {c: color_seq[i % len(color_seq)] for i, c in enumerate(dist_cats)}
        try:
            grp = anim_df.groupby(['_frame_label', dist_cat]).size().unstack(fill_value=0)
            global_max_count = int(grp.max(axis=1).max()) if not grp.empty else 0
        except Exception:
            global_max_count = 0
    else:
        dist_cats = []
        global_max_count = 0
        
    fig = make_subplots(rows=1, cols=2, column_widths=[0.7, 0.3], specs=[[{"type": "scatter"}, {"type": "bar"}]])
    first_label = frame_labels[0]
    s_trace, b_trace = make_traces_for_label(anim_df, first_label, color_by, color_map, dist_cat, dist_cats, dist_color_map, point_size, opacity)
    fig.add_trace(s_trace, row=1, col=1)
    fig.add_trace(b_trace, row=1, col=2)
    go_frames = []
    for lbl in frame_labels:
        s, b = make_traces_for_label(anim_df, lbl, color_by, color_map, dist_cat, dist_cats, dist_color_map, point_size, opacity)
        go_frames.append(go.Frame(data=[s, b], name=lbl))
    fig.frames = go_frames
    try:
        x_min = float(filtered['X'].min())
        x_max = float(filtered['X'].max())
        y_min = float(filtered['Y'].min())
        y_max = float(filtered['Y'].max())
        x_pad = (x_max - x_min) * 0.02 if x_max != x_min else 1.0
        y_pad = (y_max - y_min) * 0.02 if y_max != y_min else 1.0
        fig.update_xaxes(title_text='', row=1, col=1, range=[x_min - x_pad, x_max + x_pad], showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(title_text='', row=1, col=1, range=[y_min - y_pad, y_max + y_pad], showticklabels=False, showgrid=False, zeroline=False)
    except Exception:
        fig.update_xaxes(title_text='', row=1, col=1, showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(title_text='', row=1, col=1, showticklabels=False, showgrid=False, zeroline=False)
    
    try:
        pad = max(1, int(global_max_count * 0.02)) if global_max_count else 1
        fig.update_xaxes(title_text='count', row=1, col=2, showticklabels=True, range=[0, global_max_count + pad], showgrid=False, zeroline=False)
    except Exception:
        fig.update_xaxes(title_text='count', row=1, col=2, showticklabels=True, showgrid=False, zeroline=False)
    
    fig.update_yaxes(automargin=True, row=1, col=2, showticklabels=True, title_text='', showgrid=False)
    steps = [
        dict(method='animate', args=[[lbl], dict(mode='immediate', frame=dict(duration=int(animation_speed), redraw=True), transition=dict(duration=0))], label=lbl)
        for lbl in frame_labels
    ]
    sliders = [dict(active=0, steps=steps, x=0.0, y=-0.15, xanchor='left', yanchor='top', pad={'t': 20})]
    updatemenus = [
        dict(
            type='buttons',
            buttons=[
                dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=int(animation_speed), redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                dict(label='Pause', method='animate', args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])
            ],
            direction='left',
            pad={'r': 10, 't': 10},
            showactive=False,
            x=0.0,
            y=1.12,
            xanchor='left',
            yanchor='top'
        )
    ]
    fig.update_layout(height=700, updatemenus=updatemenus, sliders=sliders, margin=dict(t=100))
    st.plotly_chart(fig, use_container_width=True)


def main():
    df = get_data()
    controls = sidebar_controls(df)
    filtered, period_order = filter_data(df, controls)
    anim_df, frame_labels = build_animation_df(filtered, period_order, controls['window_size_input'], controls['stride_input'])
    st.title('Conspiracy Mapping')
    st.write('Use the sidebar to filter data and control the animation and styling.')
    frames_available = frame_labels if frame_labels else (sorted(anim_df['_frame_label'].unique().tolist()) if '_frame_label' in anim_df.columns else [])
    plot_dashboard(anim_df, frames_available, filtered, controls)


if __name__ == '__main__':
    main()
