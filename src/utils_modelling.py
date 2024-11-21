import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import scipy.stats as stats
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


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





class LogisticRegressionPvalues:
    """
    A custom Logistic Regression class that calculates p-values.
    """
    def __init__(self, *args, **kwargs):
        self.model = linear_model.LogisticRegression(*args, **kwargs)

    def fit(self, X, y):
        """
        Fits the model to the data and calculates p-values.
        """
        # Convert X to numpy array with float64 type
        X = np.asarray(X, dtype=np.float64)
        
        # Convert y to numpy array
        y = np.asarray(y)
        
        # Fit the base model
        self.model.fit(X, y)
        # Calculate denominator for Fisher information matrix
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom,(X.shape[1],1)).T
        F_ij = np.dot((X / denom).T,X) # Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij) # Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates # z-score for each model coefficient
        p_values = [stats.norm.sf(abs(x)) * 2 for x in z_scores] # two tailed test for p-values

        # Store results
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values_ = p_values
        



class CatImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.impute_mapping = {  # Corrigido o nome do atributo
            'mths_since_last_delinq': 'never_delinquent',
            'tot_cur_bal': 'missing',
        }
        self.missing = 'nan'

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        x_copy = x.copy()  # Corrigido para chamar o método copy()

        for feature, impute_value in self.impute_mapping.items():  # Corrigido o nome das variáveis
            x_copy[feature] = x_copy[feature].replace(self.missing, impute_value)  # Corrigido o nome das variáveis

        return x_copy


class CatCombiner(BaseEstimator, TransformerMixin):  # Corrigido o nome da classe e herança
    def __init__(self, debug=False):
        self.category_mapping = {
            'grade': [],
            'home_ownership': [['OTHER', 'NONE', 'RENT', 'ANY']],  # Corrigido o nome do campo
            'purpose': [
                ['small_business', 'educational', 'renewable_energy', 'moving'],
                ['other', 'house', 'medical', 'vacation'],
                ['wedding', 'home_improvement', 'major_purchase', 'car'],
            ],
            'addr_state': [
                ['NE', 'IA', 'NV', 'HI', 'FL'],
                ['AL', 'NM', 'NJ'],
                ['OK', 'MO', 'MD', 'NC'],
                ['AR', 'TN', 'MI', 'UT', 'VA', 'LA', 'PA', 'AZ', 'OH', 'RI', 'KY', 'DE', 'IN'],
                ['MA', 'SD', 'GA', 'MN', 'WI', 'WA', 'OR', 'IL', 'CT'],
                ['MS', 'MT', 'SC', 'VT', 'KS', 'CO', 'AK', 'NH', 'WV', 'WY', 'ID', 'DC', 'ME'],
            ],
            'initial_list_status': [],
            'verification_status': [],
            'sub_grade': [
                ['G1', 'F5', 'G5', 'G3', 'G2', 'F4', 'F3', 'G4', 'F2'],
                ['E5', 'F1', 'E4', 'E3', 'E2'],
                ['E1', 'D5', 'D4'],
                ['D3', 'D2', 'D1'],
                ['C5', 'C4', 'C3'],
                ['C2', 'C1', 'B5'],
                ['B4', 'B3'],
                ['B2', 'B1'],
                ['A5', 'A4'],
                ['A3', 'A2', 'A1']
            ],
            'term': [],
            'emp_length': [
                [1, 3],
                [4, 6],
                [7, 9]
            ],
            'inq_last_6mths': [
                [4, 33]
            ],
        }
        self.debug = debug

    def fit(self, X, y=None):
        return self

    def transform(self, x):
        x_copy = x.copy()

        for feature, category_groups in self.category_mapping.items():
            for category_group in category_groups:
                if all(isinstance(element, str) for element in category_group):
                    bundled_category = '_'.join(category_group)
                    to_replace = category_group
                else:
                    bundled_category = f'{category_group[0]}-{category_group[-1]}'  # Corrigido o acesso ao índice
                    to_replace = range(category_group[0], category_group[-1] + 1)  # Corrigido o acesso ao índice

                x_copy[feature] = x_copy[feature].replace(to_replace, bundled_category)

            x_copy[feature] = x_copy[feature].astype(str)

            if self.debug:
                print(f'Categorias de pacotes de {feature}')  # Corrigido nome da variável
                print(f'Categorias originais {x[feature].unique().tolist()}')
                print(f'Novas categorias: {x_copy[feature].unique().tolist()}')

        return x_copy


class DiscretizerCombiner(BaseEstimator, TransformerMixin):
    def __init__(self, debug=False):  # Corrigido o nome do parâmetro
        self.category_mapping = {
            'int_rate': [7, 10, 12, 14, 16, 18, 22],
            'loan_amnt': [7400, 14300, 21200, 28100],
            'dti': [4, 8, 12, 16, 20, 28],
            'annual_inc': [20000, 40000, 60000, 75000, 90000, 120000, 150000],
            'mths_since_earliest_cr_line': [151, 226, 276, 401],
            'revol_bal': [2000, 6000, 12000, 22000, 30000, 36000, 40000],
            'tot_cur_bal': [80000, 140000, 200000, 240000, 280000, 340000, 400000],
            'mths_since_last_delinq': [4, 7, 22, 37, 74],
            'open_acc': [6, 12, 21],
            'total_acc': [8, 15, 24, 36],
        }
        self.debug = debug

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        for feature, category_bins in self.category_mapping.items():
            bins = [-float('inf')] + category_bins + [float('inf')]
            labels = []

            first_bin_label = f'<={category_bins[0] / 1000:.1f}K' if category_bins[0] >= 1000 else f'<={category_bins[0]:.1f}'
            labels.append(first_bin_label)

            for i in range(1, len(category_bins)):
                lower_bound = category_bins[i-1] / 1000 if category_bins[i-1] >= 1000 else category_bins[i-1]
                upper_bound = category_bins[i] / 1000 if category_bins[i] >= 1000 else category_bins[i]
                bin_label = f'{lower_bound:.1f}K-{upper_bound:.1f}K' if category_bins[i] >= 1000 else f'{lower_bound:.1f}-{upper_bound:.1f}'
                labels.append(bin_label)

            last_bin_label = f'>{category_bins[-1] / 1000:.1f}K' if category_bins[-1] >= 1000 else f'>{category_bins[-1]:.1f}'
            labels.append(last_bin_label)

            X_copy[feature] = pd.cut(X_copy[feature], bins=bins, labels=labels, include_lowest=False, right=True)
            X_copy[feature] = X_copy[feature].astype(str)

            if self.debug:
                print(f'Discretize and bundle categories of {feature}.')
                print(f'Original range: {round(X[feature].min())} to {round(X[feature].max())}.')
                print(f'Discretized and bundled categories: {X_copy[feature].unique().tolist()}.')
                print()

        return X_copy


class CatOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Assegure-se de que há uma categoria de referência para cada feature possível
        self.reference_categories = {
            'loan_amnt': '>28.1K',
            'term': '60',
            'int_rate': '>22.0',
            'grade': 'G',
            'sub_grade': 'G1_F5_G5_G3_G2_F4_F3_G4_F2',
            'emp_length': '0',
            'home_ownership': 'OTHER_NONE_RENT_ANY',
            'annual_inc': '<=20.0K',
            'verification_status': 'Verified',
            'purpose': 'small_business_educational_renewable_energy_moving',
            'addr_state': 'NE_IA_NV_HI_FL',
            'dti': '>28.0',
            'inq_last_6mths': '4-33',
            'open_acc': '<=4.0',
            'total_acc': '<=6.0',
            'revol_bal': '<=2.0K',
            'mths_since_last_delinq': '<=8.0',
            'initial_list_status': 'f',
            'tot_cur_bal': 'missing',
            'mths_since_earliest_cr_line': '<=151.0'
        }
        self.encoder = None

    def fit(self, X, y=None):
        # Criar lista de categorias para drop baseada nas colunas presentes
        drop_categories = []
        for column in X.columns:
            ref_cat = self.reference_categories.get(column)
            if ref_cat is not None and ref_cat in X[column].unique():
                drop_categories.append(ref_cat)
            else:
                # Se não houver categoria de referência, use a primeira categoria
                first_cat = X[column].unique()[0]
                drop_categories.append(first_cat)

        # Configurar o OneHotEncoder
        self.encoder = OneHotEncoder(
            drop=drop_categories,
            sparse_output=False,
            dtype=np.int8,
            handle_unknown='ignore'
        )
        self.encoder.fit(X)
        return self

    def transform(self, X):
        if self.encoder is None:
            raise RuntimeError("The encoder must be fitted before calling transform.")
        X_one_hot = self.encoder.transform(X)
        one_hot_df = pd.DataFrame(
            X_one_hot, 
            columns=self.encoder.get_feature_names_out(),
            index=X.index
        )
        return one_hot_df

    
                
            