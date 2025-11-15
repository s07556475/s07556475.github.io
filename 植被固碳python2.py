import pandas as pd
import numpy as np
import geopandas as gpd
from libpysal import weights
import esda
import splot
from splot.esda import plot_moran, moran_scatterplot
import matplotlib.pyplot as plt
import seaborn as sns
# 移除有问题的导入，直接使用spreg
from spreg import OLS, ML_Lag, ML_Error, GM_Lag, GM_Error
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os
import warnings
warnings.filterwarnings('ignore')

# 设置工作目录
work_dir = r"C:\Users\ASUS\Desktop\shiyan"
os.chdir(work_dir)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SpatialAnalysis:
    def __init__(self, data_path, shapefile_path):
        """
        初始化空间分析类
        """
        self.data_path = data_path
        self.shapefile_path = shapefile_path
        self.data = None
        self.gdf = None
        self.w = None
        
    def load_data(self):
        """
        加载数据 - 对应Stata中的数据导入部分
        """
        try:
            # 读取Excel数据
            self.data = pd.read_excel(self.data_path)
            print(f"成功加载数据，数据形状: {self.data.shape}")
            
            # 创建ID和城市编码
            self.data['id'] = range(1, len(self.data) + 1)
            if 'name' in self.data.columns:
                self.data['city'] = pd.Categorical(self.data['name']).codes
            elif '站点名' in self.data.columns:
                self.data['city'] = pd.Categorical(self.data['站点名']).codes
                self.data.rename(columns={'站点名': 'name'}, inplace=True)
            
            # 读取shapefile
            if os.path.exists(self.shapefile_path):
                self.gdf = gpd.read_file(self.shapefile_path)
                print(f"成功加载shapefile，要素数量: {len(self.gdf)}")
            else:
                print(f"警告: 未找到shapefile文件: {self.shapefile_path}")
                # 从数据中创建模拟的地理数据
                self.gdf = self._create_simulated_gdf()
            
            print("数据加载完成")
            return self.data, self.gdf
            
        except Exception as e:
            print(f"数据加载错误: {e}")
            return None, None
    
    def _create_simulated_gdf(self):
        """
        如果缺少shapefile，创建模拟的地理数据
        """
        print("创建模拟的地理数据...")
        # 假设数据中有经纬度信息
        if 'lon' in self.data.columns and 'lat' in self.data.columns:
            geometry = gpd.points_from_xy(self.data['lon'], self.data['lat'])
        elif '经度' in self.data.columns and '纬度' in self.data.columns:
            geometry = gpd.points_from_xy(self.data['经度'], self.data['纬度'])
            self.data.rename(columns={'经度': 'lon', '纬度': 'lat'}, inplace=True)
        else:
            # 如果没有经纬度信息，创建随机坐标
            np.random.seed(42)
            n = len(self.data)
            lon = np.random.uniform(115, 125, n)  # 假设在中国东部
            lat = np.random.uniform(20, 45, n)    # 假设在中国范围
            geometry = gpd.points_from_xy(lon, lat)
            self.data['lon'] = lon
            self.data['lat'] = lat
        
        gdf = gpd.GeoDataFrame(self.data, geometry=geometry)
        return gdf
    
    def create_distance_matrix(self):
        """
        创建地理反距离矩阵 - 对应Stata中的spmat idistance
        """
        try:
            # 提取经纬度坐标
            if self.gdf is not None and 'geometry' in self.gdf.columns:
                coords = np.array([(point.x, point.y) for point in self.gdf.geometry])
            else:
                # 从数据中提取坐标
                coords = list(zip(self.data['lon'], self.data['lat']))
            
            # 创建反距离权重矩阵
            self.w = weights.DistanceBand.from_array(
                np.array(coords), 
                threshold=weights.min_threshold_distanceBand(np.array(coords)),
                alpha=-2.0,  # 反距离权重
                binary=False
            )
            
            print("地理反距离矩阵创建完成")
            return self.w
            
        except Exception as e:
            print(f"创建空间权重矩阵错误: {e}")
            return None
    
    def moran_global_analysis(self, variable, year=None):
        """
        全局莫兰指数分析 - 对应Stata中的spatgsa
        """
        try:
            if year is not None:
                data_subset = self.data[self.data['year'] == year]
            else:
                data_subset = self.data
            
            # 确保数据顺序与权重矩阵一致
            y = data_subset[variable].values
            
            # 计算莫兰指数
            moran = esda.Moran(y, self.w)
            
            print(f"全局莫兰指数分析 - {variable}" + (f" ({year}年)" if year else ""))
            print(f"莫兰指数 I: {moran.I:.4f}")
            print(f"P值: {moran.p_sim:.4f}")
            print(f"Z得分: {moran.z_sim:.4f}")
            
            return moran
            
        except Exception as e:
            print(f"莫兰指数计算错误: {e}")
            return None
    
    def moran_local_analysis(self, variable, year=None):
        """
        局部莫兰指数分析 - 对应Stata中的spatlsa
        """
        try:
            if year is not None:
                data_subset = self.data[self.data['year'] == year]
            else:
                data_subset = self.data
            
            y = data_subset[variable].values
            
            # 计算局部莫兰指数
            lisa = esda.Moran_Local(y, self.w)
            
            # 绘制莫兰散点图
            fig, ax = moran_scatterplot(lisa, aspect_equal=True)
            plt.title(f'莫兰散点图 - {variable}' + (f' ({year}年)' if year else ''))
            
            # 保存图片到工作目录
            filename = f'moran_scatter_{variable}' + (f'_{year}' if year else '') + '.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"莫兰散点图已保存: {filename}")
            plt.show()
            
            return lisa
            
        except Exception as e:
            print(f"局部莫兰指数计算错误: {e}")
            return None
    
    def yearly_moran_analysis(self, variable, start_year=2001, end_year=2020):
        """
        分年份莫兰指数分析 - 对应Stata中的循环语句
        """
        results = []
        
        for year in range(start_year, end_year + 1):
            if year in self.data['year'].unique():
                print(f"正在分析 {year} 年...")
                moran_result = self.moran_global_analysis(variable, year)
                if moran_result:
                    results.append({
                        'year': year,
                        'moran_i': moran_result.I,
                        'p_value': moran_result.p_sim,
                        'z_score': moran_result.z_sim
                    })
        
        moran_df = pd.DataFrame(results)
        
        # 保存结果到CSV
        moran_df.to_csv('yearly_moran_results.csv', index=False, encoding='utf-8-sig')
        print("分年份莫兰指数结果已保存: yearly_moran_results.csv")
        
        return moran_df
    
    def lm_test(self, dependent_var, independent_vars):
        """
        LM检验 - 对应Stata中的spatdiag
        """
        try:
            # 准备数据
            y = self.data[dependent_var].values
            X = self.data[independent_vars].values
            X = sm.add_constant(X)
            
            # OLS回归
            ols_model = OLS(y, X, w=self.w, name_y=dependent_var, name_x=['const'] + independent_vars)
            
            print("LM检验结果:")
            print(f"空间误差LM统计量: {ols_model.lm_error:.4f}")
            print(f"空间误差LM P值: {ols_model.lm_error_pval:.4f}")
            print(f"空间滞后LM统计量: {ols_model.lm_lag:.4f}")
            print(f"空间滞后LM P值: {ols_model.lm_lag_pval:.4f}")
            print(f"稳健空间误差LM统计量: {ols_model.rlm_error:.4f}")
            print(f"稳健空间误差LM P值: {ols_model.rlm_error_pval:.4f}")
            print(f"稳健空间滞后LM统计量: {ols_model.rlm_lag:.4f}")
            print(f"稳健空间滞后LM P值: {ols_model.rlm_lag_pval:.4f}")
            
            # 保存LM检验结果
            lm_results = pd.DataFrame({
                'Test': ['LM Error', 'LM Lag', 'Robust LM Error', 'Robust LM Lag'],
                'Statistic': [ols_model.lm_error, ols_model.lm_lag, ols_model.rlm_error, ols_model.rlm_lag],
                'PValue': [ols_model.lm_error_pval, ols_model.lm_lag_pval, ols_model.rlm_error_pval, ols_model.rlm_lag_pval]
            })
            lm_results.to_csv('lm_test_results.csv', index=False, encoding='utf-8-sig')
            print("LM检验结果已保存: lm_test_results.csv")
            
            return ols_model
            
        except Exception as e:
            print(f"LM检验错误: {e}")
            return None
    
    def spatial_regression(self, dependent_var, independent_vars, model_type='sdm', effects='both'):
        """
        空间回归分析 - 对应Stata中的xsmle
        """
        try:
            # 准备数据
            y = self.data[dependent_var].values
            X = self.data[independent_vars].values
            
            if model_type == 'sar':
                # 空间自回归模型
                model = ML_Lag(y, X, w=self.w, name_y=dependent_var, 
                              name_x=independent_vars)
            elif model_type == 'sem':
                # 空间误差模型
                model = ML_Error(y, X, w=self.w, name_y=dependent_var,
                               name_x=independent_vars)
            elif model_type == 'sdm':
                # 空间杜宾模型 - 需要手动创建空间滞后变量
                WX = weights.lag_spatial(self.w, X)
                X_sdm = np.hstack([X, WX])
                x_names = independent_vars + ['W_' + var for var in independent_vars]
                
                model = ML_Lag(y, X_sdm, w=self.w, name_y=dependent_var,
                              name_x=x_names)
            else:
                raise ValueError("模型类型必须是 'sar', 'sem', 或 'sdm'")
            
            # 输出结果
            print(f"\n{model_type.upper()} 模型回归结果:")
            print(model.summary)
            
            return model
            
        except Exception as e:
            print(f"空间回归错误: {e}")
            return None
    
    def lr_test(self, model1, model2):
        """
        似然比检验 - 对应Stata中的lrtest
        """
        try:
            lr_statistic = 2 * (model1.logll - model2.logll)
            df = len(model1.betas) - len(model2.betas)
            
            from scipy.stats import chi2
            p_value = 1 - chi2.cdf(lr_statistic, df)
            
            print(f"LR统计量: {lr_statistic:.4f}")
            print(f"自由度: {df}")
            print(f"P值: {p_value:.4f}")
            
            return lr_statistic, p_value
            
        except Exception as e:
            print(f"LR检验错误: {e}")
            return None, None
    
    def export_results(self, model, filename):
        """
        导出回归结果 - 对应Stata中的esttab
        """
        try:
            results_df = pd.DataFrame({
                'Variable': model.name_x,
                'Coefficient': model.betas.flatten(),
                'StdError': model.std_err.flatten(),
                'PValue': model.p_values.flatten()
            })
            
            # 添加显著性星号
            def add_stars(pval):
                if pval < 0.01:
                    return '***'
                elif pval < 0.05:
                    return '**'
                elif pval < 0.1:
                    return '*'
                else:
                    return ''
            
            results_df['Significance'] = results_df['PValue'].apply(add_stars)
            
            # 保存结果
            results_df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"结果已导出到: {filename}")
            
            return results_df
            
        except Exception as e:
            print(f"导出结果错误: {e}")
            return None

# 使用示例
def main():
    print("开始空间计量分析...")
    print(f"工作目录: {os.getcwd()}")
    
    # 初始化分析类
    spatial_analysis = SpatialAnalysis(
        data_path='data.xlsx',  # 修改为data.xlsx
        shapefile_path='point.shp'
    )
    
    # 1. 加载数据
    data, gdf = spatial_analysis.load_data()
    if data is None:
        print("数据加载失败，请检查文件路径")
        return
    
    # 显示数据基本信息
    print("\n数据基本信息:")
    print(f"数据列名: {list(data.columns)}")
    print(f"数据时间范围: {data['year'].min()} - {data['year'].max()}")
    print(f"变量数量: {len(data.columns)}")
    
    # 2. 创建空间权重矩阵
    w = spatial_analysis.create_distance_matrix()
    if w is None:
        print("空间权重矩阵创建失败")
        return
    
    # 3. 全局空间自相关分析（分年份）
    print("\n正在进行分年份莫兰指数分析...")
    moran_results = spatial_analysis.yearly_moran_analysis('npp', 2001, 2020)
    print("分年份莫兰指数结果:")
    print(moran_results)
    
    # 4. 局部空间自相关分析（示例年份）
    print("\n正在进行局部莫兰指数分析...")
    if 2001 in data['year'].unique():
        lisa_2001 = spatial_analysis.moran_local_analysis('npp', 2001)
    
    # 5. LM检验
    print("\n正在进行LM检验...")
    # 检查变量是否存在，使用可能的变量名
    independent_vars = []
    possible_vars = ['humitdy', 'rainfall', 'sunlight', 'temper', 'evpor', 'landuse',
                    '湿度', '降雨', '日照', '温度', '蒸发', '土地利用']
    
    for var in possible_vars:
        if var in data.columns:
            independent_vars.append(var)
    
    if not independent_vars:
        print("未找到独立变量，请检查数据列名")
        return
        
    print(f"使用的独立变量: {independent_vars}")
    lm_result = spatial_analysis.lm_test('npp', independent_vars)
    
    # 6. 空间回归分析
    print("\n" + "="*50)
    print("空间回归模型比较")
    print("="*50)
    
    # SDM模型
    print("\n估计SDM模型...")
    sdm_model = spatial_analysis.spatial_regression(
        'npp', independent_vars, model_type='sdm'
    )
    
    # SAR模型
    print("\n估计SAR模型...")
    sar_model = spatial_analysis.spatial_regression(
        'npp', independent_vars, model_type='sar'
    )
    
    # SEM模型
    print("\n估计SEM模型...")
    sem_model = spatial_analysis.spatial_regression(
        'npp', independent_vars, model_type='sem'
    )
    
    # 7. 导出结果
    if sdm_model:
        results = spatial_analysis.export_results(sdm_model, 'spatial_regression_results.csv')
        
        print("\n各气象因子的影响系数:")
        main_effects = results[results['Variable'].isin(independent_vars)]
        print(main_effects)
        
        # 保存主要效应结果
        main_effects.to_csv('main_effects_coefficients.csv', index=False, encoding='utf-8-sig')
        print("主要效应系数已保存: main_effects_coefficients.csv")
    
    print("\n分析完成！所有结果文件已保存到工作目录")

if __name__ == "__main__":
    main()