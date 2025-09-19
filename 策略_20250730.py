import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（兼容多种系统）
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]


class EnhancedMultiFactorStrategy:
    def __init__(self, factor_path: str, price_path: str, neutralize_path: str):
        """初始化增强版多因子策略"""
        print("\n===== 初始化策略 =====")
        # 策略参数
        self.lookback_window = 60  # 回看窗口（交易日）
        self.pred_horizon = 1  # 预测周期（交易日）
        self.transaction_cost = 0.002  # 交易成本（2‰）
        self.industry_neutral_bound = 0.03  # 行业中性约束（±3%）
        self.n_long = 50  # 做多股票数量
        self.n_short = 50  # 做空股票数量

        print(f"策略参数:")
        print(f"  回看窗口: {self.lookback_window}天")
        print(f"  预测周期: {self.pred_horizon}天")
        print(f"  交易成本: {self.transaction_cost:.2%}")
        print(f"  行业中性约束: {self.industry_neutral_bound:.2%}")
        print(f"  多空股票数量: 各{self.n_long}/{self.n_short}只")

        # 加载数据
        print("\n===== 加载数据 =====")
        self.factor_data = self._load_data(factor_path, "因子数据")
        self.price_data = self._load_data(price_path, "价格数据")
        self.neutralize_data = self._load_data(neutralize_path, "中性化数据")

        # 数据预处理
        print("\n===== 数据预处理 =====")
        self._preprocess_data()

        # 存储结果
        self.positions = None
        self.equity_curve = None
        self.performance_metrics = None

    def _load_data(self, file_path: str, data_type: str) -> pd.DataFrame:
        """加载parquet数据并返回DataFrame"""
        print(f"\n加载{data_type} - 路径: {file_path}")
        try:
            df = pd.read_parquet(file_path)
            print(f"  文件加载成功，原始形状: {df.shape}")

            # 统一日期格式（处理整数/字符串格式）
            if 'trade_date' in df.columns:
                print(f"  发现'trade_date'列，原始数据类型: {df['trade_date'].dtype}")

                # 转换为datetime（支持整数格式如20200203或字符串格式）
                if pd.api.types.is_integer_dtype(df['trade_date']):
                    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                else:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])

                # 验证日期范围
                date_min = df['trade_date'].min()
                date_max = df['trade_date'].max()
                print(f"  {data_type}日期范围: {date_min.date()} 至 {date_max.date()}")

                # 统计交易日和股票数量
                if 'ts_code' in df.columns:
                    unique_dates = df['trade_date'].nunique()
                    unique_stocks = df['ts_code'].nunique()
                    print(f"  包含{unique_dates}个交易日，{unique_stocks}只股票")
            else:
                print(f"  警告: 数据中不包含'trade_date'列")

            return df
        except Exception as e:
            print(f"  加载数据失败: {e}")
            return pd.DataFrame()

    def _preprocess_data(self):
        """数据预处理：计算收益率、对齐索引等"""
        print("\n处理价格数据:")
        print(f"  原始形状: {self.price_data.shape}")

        # 按股票和日期排序
        self.price_data = self.price_data.sort_values(['ts_code', 'trade_date'])
        print(f"  按股票和日期排序完成")

        # 计算未来pred_horizon期收益率（用于回测）
        col_name = f'next_{self.pred_horizon}d_pct_chg'
        self.price_data[col_name] = self.price_data.groupby('ts_code')['pct_chg'].shift(-self.pred_horizon) / 100  # 转为小数
        print(f"  计算未来{self.pred_horizon}日收益率完成，列名: {col_name}")

        # 检查收益率列缺失值
        missing_ratio = self.price_data[col_name].isna().mean()
        print(f"  收益率列缺失值比例: {missing_ratio:.2%}")

        # 计算基准收益率（全市场平均收益）
        self.benchmark_returns = self.price_data.groupby('trade_date')[col_name].mean().dropna()
        print(f"  计算基准收益率完成，包含{len(self.benchmark_returns)}个交易日的数据")
        print(f"  基准收益率日期范围: {self.benchmark_returns.index.min().date()} 至 {self.benchmark_returns.index.max().date()}")

    def step1_factor_orthogonalization(self, factor_cols: List[str], start_date=None, end_date=None) -> pd.DataFrame:
        """第一步：因子正交化"""
        print("\n===== 第一步：因子正交化 =====")

        # 筛选时间范围
        factor_data = self.factor_data.copy()
        print(f"  原始因子数据形状: {factor_data.shape}")
        print(f"  原始因子数据日期范围: {factor_data['trade_date'].min().date()} 至 {factor_data['trade_date'].max().date()}")

        if start_date:
            start_date = pd.to_datetime(start_date)
            factor_data = factor_data[factor_data['trade_date'] >= start_date]
            print(f"  应用start_date {start_date.date()}后形状: {factor_data.shape}")
        if end_date:
            end_date = pd.to_datetime(end_date)
            factor_data = factor_data[factor_data['trade_date'] <= end_date]
            print(f"  应用end_date {end_date.date()}后形状: {factor_data.shape}")

        # 验证筛选后的数据
        if factor_data.empty:
            print("  错误: 筛选后因子数据为空")
            return pd.DataFrame()

        print(f"  筛选后的因子数据日期范围: {factor_data['trade_date'].min().date()} 至 {factor_data['trade_date'].max().date()}")

        # 检查因子列是否存在
        missing_factors = [f for f in factor_cols if f not in factor_data.columns]
        if missing_factors:
            print(f"  警告: 因子数据中不存在以下因子: {missing_factors}")
            valid_factors = [f for f in factor_cols if f in factor_data.columns]
            if not valid_factors:
                print("  错误: 没有有效因子列，无法进行正交化")
                return pd.DataFrame()
            factor_cols = valid_factors
            print(f"  使用有效因子列: {factor_cols}")

        # 合并因子数据与中性化变量（行业、市值）
        print("\n  合并因子数据与中性化变量...")
        merged_data = pd.merge(
            factor_data[['trade_date', 'ts_code'] + factor_cols],
            self.neutralize_data[['trade_date', 'ts_code', 'total_mv', 'industry']],
            on=['trade_date', 'ts_code'],
            how='inner'
        )

        print(f"  合并后数据形状: {merged_data.shape}")
        print(f"  合并后唯一日期数: {merged_data['trade_date'].nunique()}")
        print(f"  合并后唯一股票数: {merged_data['ts_code'].nunique()}")

        # 检查合并后缺失值
        missing_cols = merged_data.columns[merged_data.isna().any()].tolist()
        if missing_cols:
            print(f"  合并后存在缺失值的列: {missing_cols}")
            for col in missing_cols:
                missing_ratio = merged_data[col].isna().mean()
                print(f"    {col}: 缺失值比例 {missing_ratio:.2%}")

        # 创建行业哑变量
        print("\n  创建行业哑变量...")
        merged_data = pd.get_dummies(merged_data, columns=['industry'], prefix='ind')
        industry_dummies = [col for col in merged_data.columns if col.startswith('ind_')]
        print(f"  创建后数据形状: {merged_data.shape}")
        print(f"  行业哑变量数量: {len(industry_dummies)}")

        # 按日期分组进行正交化
        print("\n  开始按日期进行因子正交化...")
        grouped = merged_data.groupby('trade_date')
        ortho_factors = []
        dates_processed = 0

        for date, group in grouped:
            # 打印进度
            if dates_processed % 50 == 0 and dates_processed > 0:
                print(f"    已处理{dates_processed}个交易日")

            # 对每个因子进行正交化
            for factor in factor_cols:
                # 跳过缺失值过多的日期
                if group[factor].isna().sum() > len(group) * 0.3:
                    group[factor + '_ortho'] = np.nan
                    continue

                # 准备自变量：市值和行业哑变量
                X = group[['total_mv'] + industry_dummies]
                X = X.fillna(0)  # 填充缺失值

                # 因变量：当前因子
                y = group[factor]
                mask = ~y.isna()  # 有效因子值的掩码
                X_valid = X[mask]
                y_valid = y[mask]

                # 有效数据不足时跳过
                if len(y_valid) == 0:
                    group[factor + '_ortho'] = np.nan
                    continue

                # 线性回归正交化（去除市值和行业影响）
                model = LinearRegression()
                model.fit(X_valid, y_valid)
                residuals = y_valid - model.predict(X_valid)  # 残差即为正交化因子

                # 保存正交化结果
                group.loc[mask, factor + '_ortho'] = residuals
                group.loc[~mask, factor + '_ortho'] = np.nan

            ortho_factors.append(group)
            dates_processed += 1

        print(f"  因子正交化完成，共处理{dates_processed}个交易日")

        # 合并结果并返回
        if not ortho_factors:
            print("  错误: 未生成正交化因子数据")
            return pd.DataFrame()

        result = pd.concat(ortho_factors, ignore_index=True)
        print(f"  正交化后数据形状: {result.shape}")

        # 检查正交化因子的缺失值
        for factor in factor_cols:
            col = factor + '_ortho'
            if col in result.columns:
                missing_ratio = result[col].isna().mean()
                print(f"  {col}缺失值比例: {missing_ratio:.2%}")

        return result

    def step2_remove_outliers(self, ortho_factor_cols: List[str], data: pd.DataFrame) -> pd.DataFrame:
        """第二步：去除因子极值（MAD方法）"""
        print("\n===== 第二步：去除极值 =====")
        print(f"  输入数据形状: {data.shape}")

        # 筛选有效的因子列
        valid_cols = [col for col in ortho_factor_cols if col in data.columns]
        if not valid_cols:
            print("  错误: 输入数据中不包含有效的正交化因子列")
            return data

        print(f"  处理{len(valid_cols)}个因子列: {valid_cols[:3]}...")

        # 按日期分组处理
        grouped = data.groupby('trade_date')
        cleaned_data = []
        dates_processed = 0

        for date, group in grouped:
            for col in valid_cols:
                values = group[col].copy()
                mask = ~values.isna()

                # 有效数据不足时跳过
                if mask.sum() < 30:
                    continue

                # 计算MAD（中位数绝对偏差）
                median = values[mask].median()
                mad = (values[mask] - median).abs().median()
                mad = mad if mad != 0 else 1e-10  # 避免除以0

                # 极端值替换为3倍MAD的边界
                upper_bound = median + 3 * mad
                lower_bound = median - 3 * mad
                values[mask] = values[mask].clip(lower_bound, upper_bound)

                group[col] = values

            cleaned_data.append(group)
            dates_processed += 1

        print(f"  去极值完成，共处理{dates_processed}个交易日")
        result = pd.concat(cleaned_data, ignore_index=True)
        print(f"  处理后数据形状: {result.shape}")

        return result

    def step3_standardize_factors(self, factor_cols: List[str], data: pd.DataFrame) -> pd.DataFrame:
        """第三步：因子标准化（截面标准化+滚动特征）"""
        print("\n===== 第三步：因子标准化 =====")
        print(f"  输入数据形状: {data.shape}")

        # 筛选有效的因子列
        valid_cols = [col for col in factor_cols if col in data.columns]
        if not valid_cols:
            print("  错误: 输入数据中不包含有效的因子列")
            return data

        print(f"  处理{len(valid_cols)}个因子列: {valid_cols[:3]}...")

        # 截面标准化（均值0，标准差1）
        grouped = data.groupby('trade_date')
        standardized_data = []
        dates_processed = 0

        for date, group in grouped:
            for col in valid_cols:
                values = group[col].copy()
                mask = ~values.isna()

                if mask.sum() < 30:
                    continue

                # 标准化计算
                mean = values[mask].mean()
                std = values[mask].std()
                std = std if std != 0 else 1e-10  # 避免除以0
                values[mask] = (values[mask] - mean) / std

                group[col] = values

            standardized_data.append(group)
            dates_processed += 1

        print(f"  截面标准化完成，共处理{dates_processed}个交易日")
        standardized_df = pd.concat(standardized_data, ignore_index=True)
        print(f"  标准化后数据形状: {standardized_df.shape}")

        # 计算滚动特征（因子动量，基于回看窗口）
        print(f"\n  计算因子滚动特征（回看窗口={self.lookback_window}天）...")
        final_data = []
        stocks = standardized_df['ts_code'].unique()
        total_stocks = len(stocks)

        for i, ts_code in enumerate(stocks):
            # 打印进度
            if i % 100 == 0:
                progress = i / total_stocks * 100
                print(f"    处理股票 {i}/{total_stocks} ({progress:.1f}%)")

            # 单只股票的因子数据
            stock_data = standardized_df[standardized_df['ts_code'] == ts_code].sort_values('trade_date')

            # 计算每个因子的滚动均值（因子动量）
            for col in valid_cols:
                stock_data[f'{col}_roll_mean'] = stock_data[col].rolling(
                    window=self.lookback_window,
                    min_periods=int(self.lookback_window / 2)  # 至少需要一半数据
                ).mean()

            final_data.append(stock_data)

        # 合并结果并返回
        result = pd.concat(final_data, ignore_index=True)
        print(f"  滚动特征计算完成，最终数据形状: {result.shape}")

        # 检查滚动特征的缺失值
        for col in valid_cols:
            roll_col = f'{col}_roll_mean'
            if roll_col in result.columns:
                missing_ratio = result[roll_col].isna().mean()
                print(f"  {roll_col}缺失值比例: {missing_ratio:.2%}")

        return result

    def step4_strategy_implementation(self, factor_cols: List[str],
                                        start_date: str = None, end_date: str = None,
                                        n_long: int = 50, n_short: int = 50):
            """第四步：策略实现与回测（完整逻辑）"""
            print("\n===== 第四步：策略实现与回测 =====\n")

            # 1. 转换并验证日期格式
            start_date = pd.to_datetime(start_date) if start_date else None
            end_date = pd.to_datetime(end_date) if end_date else None
            if start_date:
                print(f"  回测开始日期: {start_date.date()}")
            if end_date:
                print(f"  回测结束日期: {end_date.date()}")
            print(f"  预测周期: {self.pred_horizon}天，调仓频率: 每日")

            # 2. 因子数据预处理（正交化→去极值→标准化）
            print("\n  ==== 因子数据预处理 ====")
            # 2.1 因子正交化（去除市值和行业影响）
            ortho_factors = self.step1_factor_orthogonalization(factor_cols, start_date, end_date)
            if ortho_factors.empty:
                print("  错误: 因子正交化结果为空，终止回测")
                return

            # 2.2 去除因子极值（MAD方法）
            cleaned_factors = self.step2_remove_outliers(
                ortho_factor_cols=[f + '_ortho' for f in factor_cols],
                data=ortho_factors
            )

            # 2.3 因子标准化+滚动特征计算

            standardized_factors = self.step3_standardize_factors(
                factor_cols=[f + '_ortho' for f in factor_cols],
                data=cleaned_factors
            )

            # 3. 准备价格数据并筛选时间范围
            print("\n  ==== 价格数据准备 ====")
            price_data = self.price_data.copy()
            print(f"  原始价格数据形状: {price_data.shape}")

            # 3.1 按回测区间筛选价格数据
            if start_date:
                price_data = price_data[price_data['trade_date'] >= start_date]
            if end_date:
                price_data = price_data[price_data['trade_date'] <= end_date]
            print(f"  筛选后价格数据形状: {price_data.shape}")
            print(f"  价格数据日期范围: {price_data['trade_date'].min().date()} 至 {price_data['trade_date'].max().date()}")

            # 4. 合并因子数据与价格数据（核心数据准备）
            print("\n  ==== 合并因子与价格数据 ====")
            ret_col = f'next_{self.pred_horizon}d_pct_chg'  # 未来收益率列名
            strategy_data = pd.merge(
                standardized_factors,
                price_data[['trade_date', 'ts_code', ret_col]],  # 只保留必要列
                on=['trade_date', 'ts_code'],
                how='inner'  # 只保留两者都有的数据
            )

            # 4.1 验证合并后数据质量
            print(f"  合并后数据形状: {strategy_data.shape}")
            print(f"  合并后唯一日期数: {strategy_data['trade_date'].nunique()}")
            print(f"  合并后唯一股票数: {strategy_data['ts_code'].nunique()}")

            print(f"  未来收益率列({ret_col})缺失值比例: {strategy_data[ret_col].isna().mean():.2%}")

            # 若合并后数据为空，终止回测
            if strategy_data.empty:
                print("  错误: 因子数据与价格数据无重叠，终止回测")
                return

            # 5. 计算因子综合得分（等权加权）
            print("\n  ==== 计算因子综合得分 ====")
            # 5.1 确定参与得分计算的因子列（正交化因子+滚动特征）
            original_ortho_cols = [f + '_ortho' for f in factor_cols if f + '_ortho' in strategy_data.columns]
            roll_cols = [col for col in strategy_data.columns if any(col.startswith(f'{c}_roll_') for c in original_ortho_cols)]
            all_factor_cols = original_ortho_cols + roll_cols

            # 5.2 打印因子列信息
            print(f"  正交化因子列: {original_ortho_cols[:2]}...（共{len(original_ortho_cols)}个）")
            print(f"  滚动特征列: {roll_cols[:2]}...（共{len(roll_cols)}个）")

            # 5.3 计算综合得分（等权平均）
            strategy_data['score'] = strategy_data[all_factor_cols].mean(axis=1)

            # 5.4 检查得分列缺失值
            score_missing_ratio = strategy_data['score'].isna().mean()
            print(f"  综合得分缺失值比例: {score_missing_ratio:.2%}")
            if score_missing_ratio > 0.1:  # 缺失过多时警告
                print(f"  警告: 综合得分缺失值比例超过10%，可能影响策略效果")


            # 6. 按日期执行策略（核心回测逻辑）
            print("\n  ==== 按日期执行策略 ====")
            grouped = strategy_data.groupby('trade_date')  # 按交易日分组处理
            positions_list = []  # 存储每日持仓
            daily_returns = []   # 存储每日策略收益
            prev_weights = {}    # 上一期持仓权重（用于计算交易成本）
            dates_processed = 0  # 已处理的交易日数量
            dates_skipped = 0    # 跳过的交易日数量

            for date, group in grouped:
                # 6.1 过滤有效数据（无缺失的得分和收益率）
                valid_group = group.dropna(subset=['score', ret_col])
                if len(valid_group) < 200:  # 有效股票数量不足时跳过
                    dates_skipped += 1
                    print(f"    日期 {date.date()}: 有效股票数{len(valid_group)} < 200，跳过（累计跳过{dates_skipped}个）")
                    continue

                # 6.2 按综合得分排序（从高到低）
                sorted_group = valid_group.sort_values('score', ascending=False)

                # 6.3 组合优化（计算当日持仓权重）
                weights = self._portfolio_optimization(
                    group=sorted_group,
                    n_long=n_long,
                    n_short=n_short,
                    prev_weights=prev_weights,
                    current_date=date
                )

                if weights is None:  # 优化失败时跳过
                    prev_weights = {}
                    dates_skipped += 1
                    print(f"    日期 {date.date()}: 组合优化失败，跳过（累计跳过{dates_skipped}个）")
                    continue

                # 6.4 计算当日策略收益
                # 提取股票次日收益率（与持仓权重对应）
                stock_returns = valid_group.set_index('ts_code')[ret_col]
                # 当日收益 = 权重 × 对应股票收益率之和
                daily_return = sum(weights[stock] * stock_returns.get(stock, 0) for stock in weights)
                daily_returns.append({
                    'trade_date': date,
                    'strategy_return': daily_return
                })

                # 6.5 记录当日持仓
                pos_df = pd.DataFrame(weights.items(), columns=['ts_code', 'weight'])
                pos_df['trade_date'] = date
                positions_list.append(pos_df)

                # 6.6 更新上一期权重（用于下一日计算交易成本）
                prev_weights = weights
                dates_processed += 1

                # 6.7 打印进度（每50个交易日）
                if dates_processed % 50 == 0:
                    print(f"    已处理{dates_processed}个交易日，累计收益: {(1 + pd.DataFrame(daily_returns)['strategy_return']).cumprod().iloc[-1] - 1:.2%}")

            # 7. 整理回测结果
            print("\n  ==== 整理回测结果 ====")
            # 7.1 存储持仓数据
            if positions_list:
                self.positions = pd.concat(positions_list, ignore_index=True)
                print(f"  生成持仓数据: 共{len(self.positions)}条记录")
                print(f"  持仓日期范围: {self.positions['trade_date'].min().date()} 至 {self.positions['trade_date'].max().date()}")
            else:
                print("  警告: 未生成任何持仓数据")
                return  # 无持仓数据时终止后续流程

            # 7.2 计算净值曲线（含策略与基准）
            print("\n  ==== 计算净值曲线 ====")
            if daily_returns:
                # 转换为DataFrame并排序
                returns_df = pd.DataFrame(daily_returns).sort_values('trade_date')
                
                # 计算策略累计收益（净值曲线，不做归一化）
                returns_df['cumulative_return'] = (1 + returns_df['strategy_return']).cumprod()
                
                # 7.2.1 处理基准收益率
                # 提取回测区间内的基准收益率
                benchmark_mask = (self.benchmark_returns.index >= returns_df['trade_date'].min()) & \
                                (self.benchmark_returns.index <= returns_df['trade_date'].max())
                benchmark_in_range = self.benchmark_returns[benchmark_mask].copy()
                
                # 强制基准日期格式与策略交易日一致
                returns_df['trade_date'] = pd.to_datetime(returns_df['trade_date'], format='%Y-%m-%d')
                benchmark_in_range.index = pd.to_datetime(benchmark_in_range.index, format='%Y-%m-%d')
                
                # 对齐基准收益率（缺失值用前值填充，仍缺失则用0）
                aligned_benchmark = benchmark_in_range.reindex(
                    returns_df['trade_date'],
                    method='pad'
                ).fillna(0)
                
                # 计算基准累计收益（不做归一化）
                returns_df['benchmark_return'] = aligned_benchmark.values
                returns_df['benchmark_cumulative'] = (1 + returns_df['benchmark_return']).cumprod()
                
                # 关键修改：在计算完累计收益后，归一化到初始值为1
                returns_df['cumulative_return'] = returns_df['cumulative_return'] / returns_df['cumulative_return'].iloc[0]
                returns_df['benchmark_cumulative'] = returns_df['benchmark_cumulative'] / returns_df['benchmark_cumulative'].iloc[0]
                
                # 存储净值曲线
                self.equity_curve = returns_df
                print(f"  净值曲线计算完成，包含{len(returns_df)}个交易日")
                print(f"  策略总收益: {(returns_df['cumulative_return'].iloc[-1] - 1):.2%}")
                print(f"  基准总收益: {(returns_df['benchmark_cumulative'].iloc[-1] - 1):.2%}")
                print(f"  基准收益对齐后缺失值比例: {returns_df['benchmark_return'].isna().mean():.2%}")
            else:
                print("  警告: 未计算任何每日收益，无法生成净值曲线")
                return

            # 7.3 计算策略绩效指标
            print("\n  ==== 计算绩效指标 ====")
            self.performance_metrics = self._calculate_performance_metrics()
            print("  绩效指标计算完成")

    def _portfolio_optimization(self, group: pd.DataFrame, n_long: int, n_short: int,
                                prev_weights: Dict, current_date) -> Optional[Dict]:
        """组合优化：最大化预期收益-交易成本，带行业中性约束"""
        # 筛选多空股票池（前n_long为多，后n_short为空）
        long_candidates = group.head(n_long)['ts_code'].tolist()
        short_candidates = group.tail(n_short)['ts_code'].tolist()
        all_candidates = long_candidates + short_candidates

        # 获取行业信息（用于中性约束）
        stock_industries = self._get_stock_industries(all_candidates, current_date)
        if not stock_industries:
            print("    行业信息获取失败，无法进行行业中性约束")
            return None

        # 优化目标函数：最大化（预期收益 - 交易成本）
        def objective(weights):
            long_weights = weights[:n_long]   # 多头权重
            short_weights = weights[n_long:]  # 空头权重

            # 预期收益（多头收益 - 空头收益）
            long_score = group.set_index('ts_code').loc[long_candidates, 'score']
            short_score = group.set_index('ts_code').loc[short_candidates, 'score']
            expected_return = (long_weights @ long_score) - (short_weights @ short_score)

            # 交易成本（与上期权重的绝对偏差 × 费率）
            current_weights = {long_candidates[i]: long_weights[i] for i in range(n_long)}
            current_weights.update({short_candidates[i]: -short_weights[i] for i in range(n_short)})
            cost = self.transaction_cost * sum(abs(current_weights[stock] - prev_weights.get(stock, 0))
                                               for stock in current_weights)

            return -(expected_return - cost)  # 转为最小化问题

        # 约束条件
        constraints = [
            # 1. 多头权重和为1，空头权重和为1
            {'type': 'eq', 'fun': lambda w: sum(w[:n_long]) - 1},
            {'type': 'eq', 'fun': lambda w: sum(w[n_long:]) - 1}
        ]

        # 2. 行业中性约束（行业权重偏离≤±industry_neutral_bound）
        industries = set(stock_industries.values())
        for ind in industries:
            # 行业内的多空股票索引
            long_idx = [i for i, stock in enumerate(long_candidates) if stock_industries[stock] == ind]
            short_idx = [i for i, stock in enumerate(short_candidates) if stock_industries[stock] == ind]

            if not long_idx and not short_idx:
                continue  # 无该行业股票时跳过

            # 行业权重 = 多头权重和 - 空头权重和，约束其绝对值≤边界
            def ind_constraint(w, li=long_idx, si=short_idx):
                long_sum = sum(w[i] for i in li) if li else 0
                short_sum = sum(w[n_long + i] for i in si) if si else 0
                return self.industry_neutral_bound - abs(long_sum - short_sum)

            constraints.append({'type': 'ineq', 'fun': ind_constraint})

        # 变量边界（权重≥0，单个股票≤10%）
        bounds = [(0, 0.1) for _ in range(n_long + n_short)]

        # 初始解（等权）
        initial_guess = [1/n_long] * n_long + [1/n_short] * n_short

        # 求解优化问题
        try:
            result = minimize(
                objective,
                initial_guess,
                method='SLSQP',
                constraints=constraints,
                bounds=bounds,
                options={'maxiter': 100, 'disp': False}
            )

            if not result.success:
                print(f"    优化失败: {result.message}")
                return None

            # 整理权重结果（多头正权重，空头负权重）
            weights = {
                long_candidates[i]: result.x[i] for i in range(n_long) if result.x[i] > 1e-6
            }
            weights.update({
                short_candidates[i]: -result.x[n_long + i] for i in range(n_short) if result.x[n_long + i] > 1e-6
            })

            return weights
        except Exception as e:
            print(f"    优化异常: {e}")
            return None

    def _get_stock_industries(self, stock_list: List[str], date) -> Dict:
        """获取指定日期、指定股票的行业信息"""
        # 筛选当前日期的行业数据
        date_data = self.neutralize_data[
            (self.neutralize_data['trade_date'] == date) &
            (self.neutralize_data['ts_code'].isin(stock_list))
        ]

        # 构建股票-行业映射
        stock_industries = dict(zip(date_data['ts_code'], date_data['industry']))

        # 检查缺失的行业信息
        missing = [stock for stock in stock_list if stock not in stock_industries]
        if missing:
            print(f"警告: {len(missing)}只股票未找到行业信息")

        return stock_industries

    def _calculate_performance_metrics(self) -> Dict:
        """计算策略绩效指标"""
        if self.equity_curve is None:
            return {}
        
        # 提取收益数据
        returns = self.equity_curve['strategy_return']
        benchmark_returns = self.equity_curve['benchmark_return']
        excess_returns = returns - benchmark_returns  # 超额收益
        
        # 基本统计量
        total_return = self.equity_curve['cumulative_return'].iloc[-1] - 1
        benchmark_total = self.equity_curve['benchmark_cumulative'].iloc[-1] - 1
        days = (self.equity_curve['trade_date'].iloc[-1] - self.equity_curve['trade_date'].iloc[0]).days
        years = days / 252  # 回测年数
        
        # 年化收益率
        annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
        annual_benchmark = (1 + benchmark_total) ** (1/years) - 1 if years > 0 else 0
        
        # 年化波动率（调整周期）
        daily_vol = returns.std() * np.sqrt(1/self.pred_horizon)
        annual_vol = daily_vol * np.sqrt(252)  # 252个交易日/年
        
        benchmark_daily_vol = benchmark_returns.std() * np.sqrt(1/self.pred_horizon)
        benchmark_annual_vol = benchmark_daily_vol * np.sqrt(252)
        
        # 夏普比率（无风险收益率=0）
        sharpe = annual_return / annual_vol if annual_vol != 0 else 0
        
        # 信息比率（超额收益/超额收益波动率）
        excess_annual_mean = excess_returns.mean() * (252 / self.pred_horizon)
        excess_annual_vol = excess_returns.std() * np.sqrt(252 / self.pred_horizon)
        info_ratio = excess_annual_mean / excess_annual_vol if excess_annual_vol != 0 else 0
        
        # 最大回撤
        cumulative = self.equity_curve['cumulative_return']
        rolling_max = cumulative.cummax()  # 滚动最大值
        drawdown = (cumulative - rolling_max) / rolling_max  # 回撤率
        max_drawdown = drawdown.min()
        
        # 胜率（超额收益为正的比例）
        valid_excess = excess_returns.dropna()
        win_rate = (valid_excess > 0).mean() if len(valid_excess) > 0 else 0
        
        return {
            '总收益率': f"{total_return:.2%}",
            '基准总收益率': f"{benchmark_total:.2%}",
            '年化收益率': f"{annual_return:.2%}",
            '年化波动率': f"{annual_vol:.2%}",
            '夏普比率': f"{sharpe:.2f}",
            '信息比率': f"{info_ratio:.2f}",
            '最大回撤': f"{max_drawdown:.2%}",
            '胜率': f"{win_rate:.2%}",
            '回测周期': f"{days}天",
            '参数设置': f"回看窗口={self.lookback_window}天, 预测周期={self.pred_horizon}天"
        }

    def plot_equity_curve(self):
        """绘制策略净值曲线与基准对比"""
        if self.equity_curve is None:
            print("没有净值曲线数据可绘制")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(
            self.equity_curve['trade_date'],
            self.equity_curve['cumulative_return'],
            label='策略净值',
            linewidth=2
        )
        plt.plot(
            self.equity_curve['trade_date'],
            self.equity_curve['benchmark_cumulative'],
            label='基准净值',
            linestyle='--',
            linewidth=2
        )
        plt.title(f'策略净值曲线 vs 基准 (回看窗口={self.lookback_window}天)', fontsize=14)
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('净值（初始=1）', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        print("\n净值曲线绘制完成，显示图形窗口...")
        plt.show()

    def run(self, factor_cols: List[str] = ['alpha21', 'alpha24', 'alpha27', 'alpha28'],
            start_date: str = None, end_date: str = None):
        """运行完整策略流程"""
        print("\n===== 开始执行策略 =====")
        print(f"回测时间范围: {start_date} 至 {end_date}")
        print(f"参数设置: 回看窗口={self.lookback_window}天, 预测周期={self.pred_horizon}天")

        # 执行策略
        self.step4_strategy_implementation(factor_cols, start_date, end_date)

        # 输出绩效指标
        if self.performance_metrics:
            print("\n===== 策略绩效指标 =====")
            for key, value in self.performance_metrics.items():
                print(f"{key}: {value}")

        # 绘制净值曲线
        self.plot_equity_curve()
        print("\n===== 策略执行完成 =====")


# 使用示例
if __name__ == "__main__":
    # 初始化策略（替换为实际数据路径）
    strategy = EnhancedMultiFactorStrategy(
        factor_path="data/sliced/all_stocls_daily_factors_300_20180101_20201231",
        price_path="data/sliced/all_stocls_daily_price_vol_sliced_300_20180101_20201231",
        neutralize_path="data/sliced/neutralize_data_sliced_300_20180101_20201231"
    )

    # 运行策略（指定起止时间）
    strategy.run(
        factor_cols=['alpha21', 'alpha24', 'alpha27', 'alpha28'],
        start_date='2020-02-01',
        end_date='2020-12-30'
    )