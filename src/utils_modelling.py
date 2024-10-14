import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def analisy_univariate(data, features, histoplot=True, barplot=False, mean=None, text_y=0.5, outliers=False, kde=False, color='skyblue', figsize=(24, 12)):
    num_features = len(features)
    num_rows = num_features // 3 + (num_features % 3 > 0)

    fig, axes = plt.subplots(num_rows, 3, figsize=figsize)

    for i, feature in enumerate(features):  
        row = i // 3
        col = i % 3

        ax = axes[row, col] if num_rows > 1 else axes[col]

        if barplot:
            if mean:
                data_grouped = data.groupby([feature])[[mean]].mean().reset_index()
                data_grouped[mean] = round(data_grouped[mean], 2)
                data_grouped = data_grouped.sort_values(by=mean, ascending=True)
                value_column = mean
            else:
                data_grouped = data.groupby([feature])[[feature]].count().rename(columns={feature: 'count'}).reset_index()
                data_grouped['pct'] = round(data_grouped['count'] / data_grouped['count'].sum() * 100, 2)
                data_grouped = data_grouped.sort_values(by='pct', ascending=True)
                value_column = 'pct'

            # Lidar com cor única ou lista de cores
            if isinstance(color, list):
                colors = color
                if len(colors) < len(data_grouped):
                    colors = colors * (len(data_grouped) // len(colors) + 1)
                colors = colors[:len(data_grouped)]
            else:
                colors = [color] * len(data_grouped)
                colors[-1] = sns.color_palette(color, n_colors=2)[1]  # Versão mais escura da cor para a última barra

            bars = ax.barh(y=data_grouped[feature], width=data_grouped[value_column], color=colors)
            for index, value in enumerate(data_grouped[value_column]):
                ax.text(value + text_y, index, f'{value:.1f}{"%" if value_column == "pct" else ""}', va='center', fontsize=15)

            ax.set_yticks(ticks=range(data_grouped[feature].nunique()), labels=data_grouped[feature].tolist(), fontsize=15)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.grid(False)
            ax.get_xaxis().set_visible(False)

        elif outliers:
            sns.boxplot(data=data, x=feature, ax=ax, color=color[0] if isinstance(color, list) else color)
        else:
            sns.histplot(data=data, x=feature, kde=kde, ax=ax, color=color[0] if isinstance(color, list) else color, stat='percent')

        ax.set_title(feature)
        ax.set_xlabel('')

    if num_features < len(axes.flat):
        for j in range(num_features, len(axes.flat)):
            fig.delaxes(axes.flat[j])

    plt.tight_layout()
    return fig


def wo_discretize(df, var_discretize, target):
    # Concatenar as colunas relevantes
    df = pd.concat([df[var_discretize], target], axis=1)
    
    # Agrupar e calcular contagem e média
    df_count = df.groupby(df.columns[0], as_index=False)[df.columns[1]].count()
    df_mean = df.groupby(df.columns[0], as_index=False)[df.columns[1]].mean()
    
    # Concatenar os resultados
    df = pd.concat([df_count, df_mean[df_mean.columns[1]]], axis=1)
    
    # Renomear as colunas
    df.columns = [df.columns[0], 'n_obs', 'prop_good']
    
    # Calcular proporções e números
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    
    # Calcular WoE (Weight of Evidence)
    df['woe'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    
    # Ordenar e resetar o índice
    df = df.sort_values(['woe']).reset_index(drop=True)
    
    # Calcular diferenças
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_woe'] = df['woe'].diff().abs()
    
    # Calcular IV (Information Value)
    df['iv'] = (df['prop_n_good'] - df['prop_n_bad']) * df['woe']
    df['total_iv'] = df['iv'].sum()
    
    return df

def plot_woe(df_woe, rotation_axis=0):
        x = np.array(df_woe.iloc[: , 0].apply(str))
        y = df_woe['woe']

        plt.figure(figsize = (18, 6))
        plt.plot(x, y, marker='o', linestyle='--', color = 'k')
        plt.xlabel(df_woe.columns[0])
        plt.ylabel('Peso das evidencias')
        plt.title(str('Peso das evidencas' + df_woe.columns[0]))
        plt.xticks(rotation = rotation_axis)

def plot_horizontal_bars(df, x_col, y_col, title=None, xlabel=None, ylabel=None, 
                         color='lightcoral', figsize=(10, 6), sort=True, 
                         annotations=True, percentage=False):
    """
    Generate a horizontal bar chart from a DataFrame.
    
    Parameters:
    - df: pandas DataFrame
    - x_col: column name for x-axis values (typically numeric)
    - y_col: column name for y-axis categories
    - title: chart title (optional)
    - xlabel: x-axis label (optional)
    - ylabel: y-axis label (optional)
    - color: bar color (default 'lightcoral')
    - figsize: tuple for figure size (default (10, 6))
    - sort: boolean to sort bars by value (default True)
    - annotations: boolean to add value annotations to bars (default True)
    - percentage: boolean to format x values as percentages (default False)
    
    Returns:
    - matplotlib figure object
    """
    
    # Create a copy of the dataframe to avoid modifying the original
    plot_df = df[[y_col, x_col]].copy()
    
    # Sort if requested
    if sort:
        plot_df = plot_df.sort_values(by=x_col)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate the bars
    bars = ax.barh(plot_df[y_col], plot_df[x_col], color=color)
    
    # Add value annotations if requested
    if annotations:
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}%' if percentage else f'{width:.2f}', 
                    ha='left', va='center')
    
    # Customize the plot
    if title:
        ax.set_title(title, fontsize=14)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    
    # Format x-axis as percentage if requested
    if percentage:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
    
    # Add a grid for better readability
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adjust layout and display
    plt.tight_layout()
    
    return fig

def wo_discretize_continuos(df, var_discretize, target):
    # Concatenar as colunas relevantes
    df = pd.concat([df[var_discretize], target], axis=1)
    
    # Agrupar e calcular contagem e média
    df_count = df.groupby(df.columns[0], as_index=False)[df.columns[1]].count()
    df_mean = df.groupby(df.columns[0], as_index=False)[df.columns[1]].mean()
    
    # Concatenar os resultados
    df = pd.concat([df_count, df_mean[df_mean.columns[1]]], axis=1)
    
    # Renomear as colunas
    df.columns = [df.columns[0], 'n_obs', 'prop_good']
    
    # Calcular proporções e números
    df['prop_n_obs'] = df['n_obs'] / df['n_obs'].sum()
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    
    # Calcular WoE (Weight of Evidence)
    df['woe'] = np.log(df['prop_n_good'] / df['prop_n_bad'])

    # Calcular diferenças
    df['diff_prop_good'] = df['prop_good'].diff().abs()
    df['diff_woe'] = df['woe'].diff().abs()
    
    # Calcular IV (Information Value)
    df['iv'] = (df['prop_n_good'] - df['prop_n_bad']) * df['woe']
    df['total_iv'] = df['iv'].sum()
    
    return df
