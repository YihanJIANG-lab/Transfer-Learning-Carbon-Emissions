# ============================================================================
# Feature-engineering processor for corporate carbon-emission prediction
# Author: AI Assistant
# Date: 2025-11
# Description: GPU-accelerated feature-engineering pipeline for processing
#              corporate financial and carbon-emission data.
# ============================================================================

# Required libraries for data processing
import pandas as pd          # tabular data processing
import numpy as np           # numerical computing
import lightgbm as lgb       # gradient boosting (feature selection / modeling)
from sklearn.model_selection import train_test_split  # train/validation split
from sklearn.impute import SimpleImputer              # simple imputer (fallback)
from tqdm.auto import tqdm   # progress bars
import warnings              # warnings control
warnings.filterwarnings('ignore')  # keep logs concise

class FeatureProcessor:
    """
    Optimized feature-engineering processor (GPU-first).
    ====================================================

    Key properties:
    - Prefer GPU acceleration (LightGBM GPU) for feature selection when available.
    - Fall back to optimized CPU mode automatically if GPU is unavailable.
    - Avoid heavy GPU dependencies (e.g., RAPIDS) for broader compatibility.
    - Tailored to corporate carbon-emission prediction tasks.

    Pipeline:
    1. Initial variable screening: remove text/ID/redundant columns.
    2. Missing-value imputation: three-stage hierarchical strategy.
    3. Outlier handling: winsorization at the 1st and 99th percentiles.
    4. Feature selection: LightGBM importance ranking (Top-N).
    """
    
    def __init__(self, top_n_features=100, correlation_threshold=0.95, missing_threshold=0.3, use_gpu=True):
        """
        Initialize the processor.

        Parameters
        ----------
        top_n_features : int, default=100
            Number of features to keep. Empirically, 100 features often capture the
            dominant information while mitigating overfitting.

        correlation_threshold : float, default=0.95
            Correlation threshold. For pairs above this value, one feature is removed
            to reduce redundancy.

        missing_threshold : float, default=0.3
            Missingness threshold. Features with missing rate above this value are removed.

        use_gpu : bool, default=True
            Whether to attempt GPU acceleration. If unavailable, CPU mode is used automatically.
        """
        # Core parameters
        self.top_n_features = top_n_features
        self.correlation_threshold = correlation_threshold
        self.missing_threshold = missing_threshold
        self.use_gpu = use_gpu
        
        # Feature lists (tracked across the pipeline)
        self.initial_feature_candidates = []
        self.final_selected_features = []
        
        # Check GPU availability (LightGBM GPU only)
        self.gpu_available = self._check_gpu_support()
        
    def _check_gpu_support(self):
        """
        Check LightGBM GPU support by fitting a tiny model.

        Returns
        -------
        bool
            True  - GPU is available and LightGBM GPU mode works.
            False - GPU is unavailable or misconfigured; CPU mode will be used.
        """
        # If the user explicitly disables GPU
        if not self.use_gpu:
            print("Using CPU mode (user configuration).")
            return False
            
        try:
            # Fit a tiny model to validate GPU mode
            import lightgbm as lgb
            
            # Create a small test dataset (100 samples, 5 features)
            test_data = np.random.random((100, 5))
            test_target = np.random.random(100)
            
            # Try a GPU-mode LightGBM regressor
            lgb_model = lgb.LGBMRegressor(
                device='gpu',
                gpu_platform_id=0,
                gpu_device_id=0,
                n_estimators=10,
                verbose=-1
            )
            
            # Fit (will raise if GPU mode is not available)
            lgb_model.fit(test_data, test_target)
            
            print("✅ GPU acceleration enabled (LightGBM GPU).")
            return True
            
        except Exception as e:
            # Fall back to CPU mode on any GPU-related errors
            print(f"⚠️ GPU unavailable; falling back to optimized CPU mode: {str(e)[:100]}")
            return False

    def _initial_variable_selection(self, df):
        """
        Step 1: initial variable screening.

        We retain informative numeric features and remove:
        1) Textual fields (company name, address, etc.)
        2) Identifiers and timestamps
        3) Data-quality flags and status columns
        4) Stock-price / market variables (to avoid look-ahead bias)

        Screening rules:
        - Keep: ID columns, target variables, numeric financial metrics
        - Drop: text, categorical flags, market variables, quality-control columns

        Parameters
        ----------
        df : pd.DataFrame
            Raw input data.

        Returns
        -------
        pd.DataFrame
            Filtered data containing IDs, targets, and candidate features.
        """
        print("Step 1: starting initial variable screening...")

        # ===== Identify required columns (IDs + targets) =====
        
        # ID columns: identification and grouping (firm, year, geography, sector)
        id_cols = [col for col in ['gvkey', 'fiscalyear', 'loc', 'GICSSector'] if col in df.columns]
        
        # Raw targets: emissions across scopes (Scope 1–3 and total)
        target_cols_raw = [col for col in ['Scope1', 'Scope2', 'Scope3_upstream', 
                                          'Scope3_prod', 'Scope3_downLA', 'Scope_total'] 
                          if col in df.columns]
        
        # Log-transformed targets: mitigate skewness
        target_cols_log = [f'log_{col}' for col in target_cols_raw 
                          if f'log_{col}' in df.columns]
        
        # ===== Define drop-list (use a set for efficient lookup) =====
        cols_to_drop = set([
            # Company text fields (not informative for prediction)
            'conm', 'tic', 'cusip', 'busdesc', 'conml', 'weburl', 'companyname', 'simpleindustry',
            
            # Address fields (textual)
            'streetaddress', 'streetaddress2', 'streetaddress3', 'streetaddress4',
            'add1', 'add2', 'add3', 'add4', 'addzip', 'city', 'county', 'state', 
            'incorporation_country', 'incorporation_state',
            
            # Identifiers and contact fields
            'cik', 'ein', 'institutionid', 'companyid', 'ticker', 'periodid', 'tcprimarysectorid',
            'phone', 'fax', 'officephonevalue', 'otherphonevalue', 'officefaxvalue',
            
            # Time/status flags (may introduce leakage)
            'fyr', 'fyrc', 'dldte', 'ipodate', 'apdedate', 'fdate', 'pdate', 'periodenddate', 
            'yearfounded', 'monthfounded', 'dayfounded',
            
            # Data-quality / processing flags (internal, non-business features)
            'status', 'companytype', 'idbflag', 'prican', 'prirow', 'priusa', 'stko', 
            'acctchg', 'acctstd', 'acqmeth', 'bspr', 'compst', 'final', 'ltcm', 'ogm', 
            'scf', 'stalt', 'udpl', 'upd',
            
            # Audit/management-related information (non-financial)
            'au', 'auop', 'auopic', 'ceoso', 'cfoso', 'rank',
            
            # Duplicate or redundant fields
            'trucost total revenue',
            
            # Stock price / market variables (avoid look-ahead bias)
            'ajex', 'ajp', 'mkvalt', 'prcc_c', 'prcc_f', 'prch_c', 'prch_f', 
            'prcl_c', 'prcl_f', 'cshtr_c', 'cshtr_f', 'dvpsp_c', 'dvpsp_f', 
            'dvpsx_c', 'dvpsx_f'
        ])
        
        # Drop di_* columns (typically data-availability flags)
        di_cols = {col for col in df.columns if col.startswith('di_')}
        cols_to_drop.update(di_cols)
        
        # Define required columns to keep (IDs + targets)
        keep_cols = set(id_cols + target_cols_raw + target_cols_log)
        
        # ===== Screen numeric features =====
        
        # Only these dtypes are treated as numeric features
        numeric_dtypes = ['float64', 'int64', 'float32', 'int32']
        potential_features = []  # candidate features
        
        # Traverse columns and collect candidate numeric features
        for col in df.columns:
            if (col not in keep_cols and
                col not in cols_to_drop and
                df[col].dtype in numeric_dtypes):
                potential_features.append(col)

        # Save screened candidate features
        self.initial_feature_candidates = potential_features
        
        # Build final retained columns: required + candidates
        retained_cols = list(keep_cols) + potential_features
        
        # Safety check: retain only existing columns
        retained_cols = [col for col in retained_cols if col in df.columns]
        
        # Create a copy (avoid modifying the original data)
        df_filtered = df[retained_cols].copy()
        
        # Report screening statistics
        print(
            f"Initial screening completed. Retained {len(retained_cols)} columns "
            f"(including IDs and targets) out of {len(df.columns)} total columns."
        )
        print(f"Identified {len(self.initial_feature_candidates)} numeric candidate features for downstream processing.")
        
        return df_filtered

    def _impute_missing_values(self, df):
        """
        Step 2: hierarchical missing-value imputation.

        We use a three-stage imputation strategy to exploit the hierarchical structure
        of the data:
        1) Firm-level imputation: median within the same firm across time.
        2) Sector-year imputation: median within the same sector and fiscal year.
        3) Global imputation: median over the full dataset (fallback).

        Median is used for robustness against outliers.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset containing candidate features.

        Returns
        -------
        pd.DataFrame
            Imputed dataset.
        """
        print("\nStep 2: starting hierarchical missing-value imputation...")
        
        # Only process candidate features (do not touch IDs/targets)
        features_to_impute = [col for col in self.initial_feature_candidates if col in df.columns]
        
        # Safety check
        if not features_to_impute:
            print("No features require imputation.")
            return df
        
        # ===== Stage 1: firm-level median imputation =====
        print("Stage 1: firm-level median imputation...")
        
        # Pre-compute firm-level medians for efficiency
        company_medians = df.groupby('gvkey')[features_to_impute].median()
        
        # Impute per feature (progress bar for monitoring)
        for col in tqdm(features_to_impute, desc="Firm-level imputation"):
            df[col] = df[col].fillna(df['gvkey'].map(company_medians[col]))
        
        # Stage-1 summary
        missing_after_step1 = df[features_to_impute].isnull().sum().sum()
        print(f"After Stage 1, remaining missing values: {missing_after_step1}.")

        # ===== Stage 2: sector-year median imputation =====
        if missing_after_step1 > 0:
            print("Stage 2: sector-year median imputation...")
            
            # Compute sector-year medians
            sector_year_medians = df.groupby(['GICSSector', 'fiscalyear'])[features_to_impute].median()
            
            # Impute per feature
            for col in tqdm(features_to_impute, desc="Sector-year imputation"):
                for (sector, year), median_val in sector_year_medians[col].items():
                    mask = (df['GICSSector'] == sector) & (df['fiscalyear'] == year) & df[col].isnull()
                    df.loc[mask, col] = median_val
        
        # Stage-2 summary
        missing_after_step2 = df[features_to_impute].isnull().sum().sum()
        print(f"After Stage 2, remaining missing values: {missing_after_step2}.")
        
        # ===== Stage 3: global median imputation =====
        if missing_after_step2 > 0:
            print("Stage 3: global median imputation (fallback)...")
            
            # Compute global medians
            overall_medians = df[features_to_impute].median()
            
            # Fill remaining missing values
            df[features_to_impute] = df[features_to_impute].fillna(overall_medians)
            
            print(f"Stage 3 filled the remaining {missing_after_step2} missing values.")
            
        print("✅ Missing-value imputation completed.")
        return df

    def _handle_outliers(self, df):
        """
        Step 3: robust outlier handling (winsorization).

        We winsorize at the 1st and 99th percentiles. Compared with dropping outliers,
        winsorization preserves sample size and the main distributional structure while
        reducing the influence of extreme values.

        Parameters
        ----------
        df : pd.DataFrame
            Imputed dataset.

        Returns
        -------
        pd.DataFrame
            Dataset after outlier handling.
        """
        print("\nStep 3: handling outliers (winsorizing at 1% and 99%)...")
        
        # Features to winsorize
        features_to_winsorize = [col for col in self.initial_feature_candidates if col in df.columns]
        
        # Safety check
        if not features_to_winsorize:
            print("No features require outlier handling.")
            return df
        
        # ===== Pre-compute percentile cutoffs =====
        print("Computing winsorization cutoffs...")
        percentiles_01 = df[features_to_winsorize].quantile(0.01)
        percentiles_99 = df[features_to_winsorize].quantile(0.99)
        
        # ===== Apply winsorization =====
        print("Applying winsorization...")
        
        for col in tqdm(features_to_winsorize, desc="Winsorizing"):
            df[col] = np.clip(df[col], percentiles_01[col], percentiles_99[col])
            
        print("✅ Outlier handling completed.")
        return df

    def _automated_feature_selection(self, df):
        """
        Step 4: automated feature selection
        ==================
        
        Description:
        --------
        Multi-stage feature selection to identify the most informative Top-N features from
        hundreds/thousands of candidates. This implementation is tailored to carbon-emission
        prediction and uses ``log_Scope_total`` as the primary target for ranking.
        
        Strategy
        --------
        4a) Quality filter: remove features with high missingness and zero variance.
        4b) Correlation filter: remove highly correlated redundant features.
        4c) Importance ranking: select Top-N features by LightGBM importance.
        
        Benefits
        --------
        - Combines statistical filtering with model-based selection.
        - Supports GPU-accelerated training for large-scale data.
        - Captures non-linear relationships between features and the target.
        
        Parameters
        ----------
        df : pd.DataFrame
            Preprocessed dataset.
            
        Returns
        -------
        pd.DataFrame
            Dataset with the selected feature set recorded in ``self.final_selected_features``.
        """
        print("\nStep 4: starting automated feature selection...")
        
        # Get currently available candidate features
        features = [col for col in self.initial_feature_candidates if col in df.columns]
        
        # Safety check
        if not features:
            print("❌ No usable features available for selection.")
            self.final_selected_features = []
            return df
        
        # ===== 4a: Quality filtering =====
        print(f"  - 4a: Quality filter — removing features with missing rate > {self.missing_threshold*100:.1f}% and zero variance...")
        
        good_features = []  # features that pass the quality checks
        
        # Per-feature quality checks
        for col in features:
            # Missing rate = #missing / #samples
            missing_rate = df[col].isnull().sum() / len(df)
            
            # Unique-value count: proxy for non-zero variance
            unique_count = df[col].nunique()
            
            # Criteria: missing rate below threshold and non-constant column
            if missing_rate < self.missing_threshold and unique_count > 1:
                good_features.append(col)
        
        # Update feature list and report
        features = good_features
        print(f"    ✅ Quality filter completed; remaining: {len(features)} high-quality features.")
        
        # Check whether any features remain
        if not features:
            print("❌ No features remain after quality filtering.")
            self.final_selected_features = []
            return df
        
        # ===== 4b: Correlation filtering =====
        # Goal: remove highly correlated redundant features to mitigate multicollinearity
        if len(features) > 1 and len(features) <= 1000:  # compute correlations only at a reasonable scale
            print(f"  - 4b: Correlation filter — removing redundant features with |corr| > {self.correlation_threshold}...")
            
            # Compute absolute correlation matrix
            corr_matrix = df[features].corr().abs()
            
            # Keep upper triangle to avoid duplicate pairs
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Identify features to drop
            to_drop = [column for column in upper.columns 
                      if any(upper[column] > self.correlation_threshold)]
            
            # Remove redundant features
            features = [f for f in features if f not in to_drop]
            
            print(f"    ✅ Correlation filter completed; dropped {len(to_drop)} redundant features; remaining: {len(features)}.")
            
        elif len(features) > 1000:
            print(f"  - 4b: ⚠️ Too many features ({len(features)}); skipping correlation filtering to avoid memory pressure.")
        
        # ===== 4c: LightGBM importance ranking =====
        print(f"  - 4c: LightGBM selection — ranking Top-{self.top_n_features} most important features...")
        
        # Primary target for feature ranking (log-transformed total emissions)
        target_col = 'log_Scope_total'
        
        # ===== Target verification =====
        if target_col not in df.columns:
            print(f"    ❌ Error: target column '{target_col}' does not exist in the DataFrame.")
            print(f"    Available log_* columns: {[col for col in df.columns if col.startswith('log_')]}")
            
            # Fallback: use another emissions-related log_* target if available
            backup_targets = [col for col in df.columns if 'log_' in col and 'scope' in col.lower()]
            if backup_targets:
                target_col = backup_targets[0]
                print(f"    Using fallback target column: {target_col}")
            else:
                print("    ⚠️ No suitable target found; falling back to the first N features.")
                self.final_selected_features = features[:self.top_n_features]
                return df
        
        print(f"    Target column for ranking: {target_col}")
        
        # ===== Training data preparation =====
        # Use complete cases for (target + selected feature set)
        temp_df = df.dropna(subset=[target_col] + features).copy()
        
        # Complete-case check
        if len(temp_df) == 0:
            print("    ❌ Warning: no complete training data (all rows have missing values).")
            self.final_selected_features = features[:self.top_n_features]
            return df
        
        print(f"    Complete training samples: {len(temp_df):,}")
        
        # Train on all available complete cases (no subsampling)
        print(f"    Training on all {len(temp_df):,} samples (no subsampling).")
            
        # ===== Build training matrices =====
        X = temp_df[features]
        y = temp_df[target_col]
        
        print(f"    Train shapes: X={X.shape}, y={y.shape}")
        print(f"    Target stats ({target_col}): min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}")

        # ===== LightGBM parameter configuration =====
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'device': 'gpu' if self.gpu_available else 'cpu',
            'random_state': 42,
            'verbose': -1,
            
            # Model complexity
            'n_estimators': 100,
            'num_leaves': 31,
            'learning_rate': 0.1,
            
            # Regularization / randomness (reduce overfitting)
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
        }
        
        # GPU-specific parameters
        if self.gpu_available:
            lgb_params.update({
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            })
        
        # ===== Train LightGBM =====
        try:
            mode = "GPU" if self.gpu_available else "CPU"
            print(f"    Training LightGBM in {mode} mode...")
            
            # Create regressor
            lgb_model = lgb.LGBMRegressor(**lgb_params)
            
            # Fit with early stopping
            lgb_model.fit(
                X, y,
                eval_set=[(X, y)],
                eval_metric='rmse',
                callbacks=[
                    lgb.early_stopping(stopping_rounds=10, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            # ===== Extract feature importances =====
            feature_importances = pd.Series(lgb_model.feature_importances_, index=features)
            print(f"    Importance stats: min={feature_importances.min():.4f}, max={feature_importances.max():.4f}")
            
            # Keep Top-N features by importance
            top_features_sorted = feature_importances.sort_values(ascending=False).head(self.top_n_features)
            
            # Save importance scores and names for downstream analysis
            self.feature_importance_ = top_features_sorted.values
            
            self.feature_names_ = top_features_sorted.index.tolist()
            
            # Final selected features
            self.final_selected_features = top_features_sorted.index.tolist()
            
            print(f"    ✅ Selected Top-{len(self.final_selected_features)} features successfully.")
            
            top_10_features = feature_importances.sort_values(ascending=False).head(10)
            print("    Top-10 feature preview:")
            for idx, (feature, importance) in enumerate(top_10_features.items(), 1):
                print(f"      {idx:2d}. {feature:<30} importance: {importance:.4f}")
            
        except Exception as e:
            # ===== Fallback: LightGBM training failed =====
            print(f"    ❌ LightGBM training failed: {str(e)[:100]}...")
            print("    Falling back to selecting the first N features.")
            
            # Fallback strategy: select the first N features in the current order
            self.final_selected_features = (features[:self.top_n_features] 
                                          if len(features) > self.top_n_features 
                                          else features)
        
        # ===== Summary =====
        print("Feature selection completed.")
        print(f"   Selected features: {len(self.final_selected_features)}")
        print(f"   Target column: {target_col}")
        print("   Ready for downstream carbon-emission prediction modeling.")
        
        return df

    def process(self, df_raw):
        """
        Run the complete feature-engineering pipeline.

        This is the main entry point. It executes four sequential steps:
        1) Initial variable screening.
        2) Hierarchical missing-value imputation.
        3) Robust outlier handling (winsorization).
        4) Automated feature selection (LightGBM Top-N).

        Data flow:
        raw data → screening → imputation → outlier handling → feature selection → final feature set

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw corporate financial and carbon-emission dataset.

        Returns
        -------
        pd.DataFrame
            Final dataset containing IDs, targets, and the selected feature set.
        """
        print("🚀" + "="*60)
        print("   Starting feature-engineering pipeline")
        print(f"   Mode: {'GPU-accelerated' if self.gpu_available else 'CPU'}")
        print("🚀" + "="*60)
        
        # ===== Pre-check =====
        if 'log_Scope_total' not in df_raw.columns:
            print("⚠️ Warning: the dataset does not contain the primary target 'log_Scope_total'.")
            available_log_cols = [col for col in df_raw.columns if col.startswith('log_')]
            print(f"   Available log_* columns: {available_log_cols}")
        
        # ===== Create a working copy (do not modify raw input) =====
        df = df_raw.copy()
        
        # ===== Execute pipeline =====
        print(f"\nInput shape: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
        
        df = self._initial_variable_selection(df)    # Step 1: screening
        df = self._impute_missing_values(df)         # Step 2: imputation
        df = self._handle_outliers(df)               # Step 3: outlier handling
        df = self._automated_feature_selection(df)   # Step 4: feature selection
        
        # ===== Build final output =====
        print("\nBuilding final feature-engineering output...")
        
        # ID columns (for tracking and analysis)
        id_cols = [col for col in ['gvkey', 'fiscalyear', 'loc', 'GICSSector'] 
                  if col in df.columns]
        
        # Raw carbon-emission target variables (original scale)
        target_cols_original = [col for col in df.columns 
                               if col in ['Scope1', 'Scope2', 'Scope3_upstream', 
                                         'Scope3_prod', 'Scope3_downLA', 'Scope_total']]
        
        # Log-transformed target variables
        target_cols_log = [col for col in df.columns 
                          if col.startswith('log_') and 'scope' in col.lower()]
        
        # Merge all target variables
        all_target_cols = target_cols_original + target_cols_log
        
        # Final columns: IDs + targets + selected features
        final_cols = id_cols + all_target_cols + self.final_selected_features
        
        # Safety check: keep only existing columns
        final_cols = [col for col in final_cols if col in df.columns]

        # ===== Output summary =====
        print("\nFinal dataset summary:")
        print(f"   ID columns: {len(id_cols)} -> {id_cols}")
        print(f"   Raw target columns: {len(target_cols_original)} -> {target_cols_original}")
        print(f"   Log target columns: {len(target_cols_log)} -> {target_cols_log}")
        print(f"   Selected feature columns: {len(self.final_selected_features)}")
        print(f"   Total columns: {len(final_cols)}")
        print(f"   Total samples: {df.shape[0]:,}")
        
        # ===== Primary target verification =====
        if 'log_Scope_total' in final_cols:
            print("   ✅ Primary target 'log_Scope_total' is included.")
        else:
            print("   ⚠️ Warning: primary target 'log_Scope_total' is NOT included in the output.")
        
        print("🎉" + "="*60)
        print("   Feature-engineering pipeline completed.")
        print("   Data are ready for downstream machine-learning modeling.")
        print("🎉" + "="*60)
        
        return df[final_cols]