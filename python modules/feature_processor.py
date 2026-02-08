# feature_processor.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

class FeatureProcessor:
    """
    A wrapper class that implements a complete feature-engineering pipeline, designed to
    produce a reusable, high-quality core feature set for corporate carbon-emission prediction.

    Pipeline:
    initial variable screening -> missing-value imputation -> outlier handling -> automated feature selection.
    """
    def __init__(self, top_n_features=100, correlation_threshold=0.95, missing_threshold=0.5):
        """
        Initialize the processor.

        Args:
            top_n_features (int): Number of features to keep after LightGBM importance ranking.
            correlation_threshold (float): Threshold for dropping highly correlated features.
            missing_threshold (float): Threshold for dropping features with high missingness.
        """
        self.top_n_features = top_n_features
        self.correlation_threshold = correlation_threshold
        self.missing_threshold = missing_threshold
        self.initial_feature_candidates = []
        self.final_selected_features = []

    def _initial_variable_selection(self, df):
        """
        Step 1: initial variable screening based on definitions and domain heuristics.

        The goal is to remove obviously non-informative, redundant, or hard-to-process variables.
        """
        print("Step 1: starting initial variable screening...")

        # Identify and retain key ID/time/category variables
        id_cols = ['gvkey', 'fiscalyear', 'loc', 'GICSSector']
        target_cols_raw = ['Scope1', 'Scope2', 'Scope3_upstream', 'Scope3_prod', 'Scope3_downLA', 'Scope_total']
        target_cols_log = [f'log_{col}' for col in target_cols_raw]
        
        # Identify columns to drop
        cols_to_drop = [
            # 1. Textual descriptors that cannot be directly used for modeling
            'conm', 'tic', 'cusip', 'busdesc', 'conml', 'weburl', 'companyname', 'simpleindustry',
            'streetaddress', 'streetaddress2', 'streetaddress3', 'streetaddress4',
            # 2. Detailed address fields (higher-level categories are already captured)
            'add1', 'add2', 'add3', 'add4', 'addzip', 'city', 'county', 'state', 'incorporation_country', 'incorporation_state',
            # 3. Miscellaneous IDs/codes (gvkey is already the unique firm identifier)
            'cik', 'ein', 'institutionid', 'companyid', 'ticker', 'periodid', 'tcprimarysectorid',
            # 4. Contact information
            'phone', 'fax', 'officephonevalue', 'otherphonevalue', 'officefaxvalue',
            # 5. Date-related columns (fiscalyear is our primary time unit)
            'fyr', 'fyrc', 'dldte', 'ipodate', 'apdedate', 'fdate', 'pdate', 'periodenddate', 'yearfounded', 'monthfounded', 'dayfounded',
            # 6. Status/flags/categorical codes with low information content or redundant with sector variables
            'status', 'companytype', 'idbflag', 'prican', 'prirow', 'priusa', 'stko', 'acctchg', 'acctstd', 'acqmeth',
            'bspr', 'compst', 'final', 'ltcm', 'ogm', 'scf', 'stalt', 'udpl', 'upd',
            # 7. Audit/SOX-related information
            'au', 'auop', 'auopic', 'ceoso', 'cfoso', 'rank',
            # 8. Trucost cost/ratio/disclosure scores (avoid leakage; we only use financial data to predict emissions)
            'trucost total revenue',  # may overlap with Compustat revenue
            # ... drop non-target variables with di_ prefix
            'di_319380', 'di_319381', 'di_319382', 'di_319383', 'di_319384', 'di_319385', 'di_319404', 
            'di_319405', 'di_319406', 'di_319407', 'di_319408', 'di_319409', 'di_368314', 'di_326738', 
            'di_368754', 'di_368742', 'di_319437', 'di_319438', 'di_319439', 'di_319440', 'di_319441', 
            'di_319442', 'di_319541', 'di_319542', 'di_319545', 'di_319546', 'di_319547', 'di_319548',
            'di_319549', 'di_319550', 'di_319552', 'di_319553', 'di_319554', 'di_319557', 'di_319558',
            'di_319559', 'di_319560', 'di_319562', 'di_319563', 'di_319564', 'di_319565', 'di_319566',
            'di_319568', 'di_319569', 'di_319469', 'di_319467', 'di_319465', 'di_319466', 'di_319468',
            'di_319470', 'di_319416', 'di_319555', 'di_319570', 'di_329704',
            # 9. Text disclosure information
            'di_319403_text', 'di_329689_text', 'di_329692_text', 'di_329697_text', 'di_329701_text', 'di_368758_text',
            # 10. Price-related / adjustment-factor variables, or highly redundant fiscal/calendar duplicates
            'ajex', 'ajp', 'mkvalt', 'prcc_c', 'prcc_f', 'prch_c', 'prch_f', 'prcl_c', 'prcl_f', 'cshtr_c', 'cshtr_f',
            'dvpsp_c', 'dvpsp_f', 'dvpsx_c', 'dvpsx_f'
        ]
        
        # Ensure required columns are not dropped
        keep_cols = id_cols + target_cols_raw + target_cols_log
        cols_to_drop = [col for col in cols_to_drop if col in df.columns and col not in keep_cols]
        
        # Screen candidate numeric features
        potential_features = [col for col in df.columns if col not in keep_cols and col not in cols_to_drop and (df[col].dtype in ['float64', 'int64', 'float32', 'int32'])]

        self.initial_feature_candidates = potential_features
        retained_cols = keep_cols + potential_features
        df_filtered = df[retained_cols].copy()
        
        print(
            f"Initial screening completed. Retained {len(retained_cols)} columns "
            f"(including IDs and targets) out of {len(df.columns)} total columns."
        )
        print(f"Identified {len(self.initial_feature_candidates)} numeric candidate features for downstream processing.")
        return df_filtered

    def _impute_missing_values(self, df):
        """
        Step 2: two-stage missing-value imputation for numeric features.
        """
        print("\nStep 2: starting missing-value imputation...")
        features_to_impute = self.initial_feature_candidates
        
        # Use tqdm to display progress
        tqdm.pandas(desc="Stage 1 (firm-level median)")
        # Stage 1: firm-level time-series median imputation
        df[features_to_impute] = df.groupby('gvkey')[features_to_impute].progress_apply(lambda x: x.fillna(x.median()))
        
        missing_after_step1 = df[features_to_impute].isnull().sum().sum()
        print(f"After Stage 1, remaining missing values: {missing_after_step1}.")

        if missing_after_step1 > 0:
            tqdm.pandas(desc="Stage 2 (sector-year median)")
            # Stage 2: sector-year median imputation for remaining missing values
            df[features_to_impute] = df.groupby(['GICSSector', 'fiscalyear'])[features_to_impute].progress_apply(lambda x: x.fillna(x.median()))
        
        missing_after_step2 = df[features_to_impute].isnull().sum().sum()
        print(f"After Stage 2, remaining missing values: {missing_after_step2}.")
        
        if missing_after_step2 > 0:
            # Final safeguard: fill any remaining missing values with zeros
            df[features_to_impute] = df[features_to_impute].fillna(0)
            print(f"Filled the remaining {missing_after_step2} missing values with zeros.")
            
        print("Missing-value imputation completed.")
        return df

    def _handle_outliers(self, df):
        """
        Step 3: outlier handling for numeric features (winsorization).
        """
        print("\nStep 3: handling outliers (winsorizing at 1% and 99%)...")
        
        features_to_winsorize = self.initial_feature_candidates
        
        for col in tqdm(features_to_winsorize, desc="Winsorizing"):
            p1 = df[col].quantile(0.01)
            p99 = df[col].quantile(0.99)
            df[col] = np.clip(df[col], p1, p99)
            
        print("Outlier handling completed.")
        return df

    def _automated_feature_selection(self, df):
        """
        Step 4: automated feature selection in three sub-steps.
        """
        print("\nStep 4: starting automated feature selection...")
        
        features = self.initial_feature_candidates
        
        # 4a: remove features with high missingness and zero variance
        print(f"  - 4a: filtering features with missingness > {self.missing_threshold*100:.1f}% and zero variance...")
        good_features = []
        for col in features:
            if df[col].isnull().sum() / len(df) < self.missing_threshold and df[col].nunique() > 1:
                good_features.append(col)
        print(f"    Remaining features after filtering: {len(good_features)}.")
        features = good_features
        
        # 4b: remove highly correlated features
        print(f"  - 4b: removing highly correlated features (>{self.correlation_threshold})...")
        corr_matrix = df[features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
        features = [f for f in features if f not in to_drop]
        print(f"    Dropped {len(to_drop)} highly correlated features; remaining: {len(features)}.")
        
        # 4c: LightGBM importance ranking
        print(f"  - 4c: selecting Top-{self.top_n_features} features via LightGBM importance ranking...")
        # We use the most informative target 'log_Scope_Total' for ranking.
        temp_df = df.dropna(subset=['log_Scope_Total'] + features).copy()
        
        # If the dataset is very large, subsample for efficiency
        if len(temp_df) > 200000:
            temp_df = temp_df.sample(n=200000, random_state=42)
            
        X = temp_df[features]
        y = temp_df['log_Scope_Total']

        lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
        lgb_model.fit(X, y)
        
        feature_importances = pd.Series(lgb_model.feature_importances_, index=features)
        self.final_selected_features = feature_importances.sort_values(ascending=False).head(self.top_n_features).index.tolist()
        
        print(f"Feature selection completed. Selected {len(self.final_selected_features)} core features.")
        return df

    def process(self, df_raw):
        """
        Run the complete feature-engineering pipeline.
        
        Args:
            df_raw (pd.DataFrame): Raw (merged) dataset.
        
        Returns:
            pd.DataFrame: Processed dataset containing only the final features and targets.
        """
        # Copy to avoid modifying the original DataFrame
        df = df_raw.copy()
        
        # Execute pipeline
        df = self._initial_variable_selection(df)
        df = self._impute_missing_values(df)
        df = self._handle_outliers(df)
        df = self._automated_feature_selection(df)
        
        # Build final output DataFrame
        id_cols = ['gvkey', 'fiscalyear', 'loc', 'GICSSector']
        target_cols_original = ['Scope1', 'Scope2', 'Scope3_upstream', 'Scope3_prod', 'Scope3_downLA', 'Scope_total']
        target_cols_log = ['log_Scope1', 'log_Scope2', 'log_Scope3_upstream', 'log_Scope3_prod', 'log_Scope3_downLA', 'log_Scope_total']
        
        # Include both raw and log-transformed targets
        all_target_cols = target_cols_original + target_cols_log
        final_cols = id_cols + all_target_cols + self.final_selected_features
        
        # Ensure all selected columns exist
        final_cols = [col for col in final_cols if col in df.columns]

        print("\nFinal output DataFrame contains:")
        print(f"  - ID columns: {len(id_cols)}")
        print(f"  - Raw target columns: {len([col for col in target_cols_original if col in df.columns])}")
        print(f"  - Log-transformed target columns: {len([col for col in target_cols_log if col in df.columns])}") 
        print(f"  - Feature columns: {len(self.final_selected_features)}")
        print(f"  - Total columns: {len(final_cols)}")
        
        return df[final_cols]