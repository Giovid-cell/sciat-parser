"""
SC-IAT Parser for MinnoJS/Qualtrics Data
==========================================
Parses embedded SC-IAT data from Qualtrics surveys built with MinnoJS.
Produces trial-level output for D-score calculation and exclusion criteria.

Author: Parser for Selvaticə's dissertation research
Usage:
    python sciat_minnojs_parser.py input.csv [output_prefix] [--iat-column SC_IAT_aiuto] [--id-column ResponseId]

Output:
    - {prefix}_trial_level.csv: All trials with metadata
    - {prefix}_summary.csv: Per-participant summary statistics
    - {prefix}_quality_report.csv: Data quality flags for exclusions
"""

import pandas as pd
import numpy as np
import csv
import sys
import re
import argparse
from io import StringIO
from pathlib import Path
from typing import Optional, Tuple, List, Dict


class SCIATParser:
    """
    Parser for SC-IAT data embedded in Qualtrics exports from MinnoJS experiments.
    
    MinnoJS exports SC-IAT data as a nested CSV within a single Qualtrics column.
    This parser handles:
    - Double-double quote escaping ("")
    - Internal commas within quoted fields
    - Mixed line endings
    - Multiple participants in one file
    """
    
    def __init__(
        self, 
        input_file: str, 
        iat_column: str = 'SC_IAT_aiuto',
        id_column: str = 'ResponseId',
        skip_rows: int = 2  # Qualtrics exports have 2 header rows to skip
    ):
        """
        Initialize the parser.
        
        Args:
            input_file: Path to Qualtrics CSV export
            iat_column: Column name containing embedded SC-IAT data
            id_column: Column name containing participant IDs
            skip_rows: Number of Qualtrics metadata rows to skip (typically 2)
        """
        self.input_file = Path(input_file)
        self.iat_column = iat_column
        self.id_column = id_column
        self.skip_rows = skip_rows
        
        self.raw_data: Optional[pd.DataFrame] = None
        self.trial_data: Optional[pd.DataFrame] = None
        self.summary_data: Optional[pd.DataFrame] = None
        self.quality_report: Optional[pd.DataFrame] = None
        
        # SC-IAT configuration
        self.practice_blocks = [1, 3]
        self.critical_blocks = [2, 4]
        self.end_block = 9
        
        # D-score calculation parameters (Karpinski & Steinman, 2006)
        self.rt_upper_limit = 1500  # Timeout threshold
        self.rt_lower_limit = 350   # Anticipatory response threshold
        self.max_error_rate = 0.40  # Maximum allowable error rate
        self.min_valid_trials = 29  # Minimum trials required
        self.max_timeout_rate = 0.0833  # ~8.33% timeout rate
        
    def load_data(self) -> 'SCIATParser':
        """Load and filter raw Qualtrics data."""
        print(f"📂 Loading data from: {self.input_file}")
        
        try:
            # Qualtrics exports have:
            # Row 0: Column names (headers)
            # Row 1: Question text (skip)
            # Row 2: ImportId metadata (skip)
            # Row 3+: Actual data
            # Using sep=None and engine='python' for better delimiter detection
            self.raw_data = pd.read_csv(
                self.input_file, 
                dtype=str, 
                encoding='utf-8-sig',
                sep=None,
                engine='python',
                header=0,  # First row is header
                skiprows=[1, 2] if self.skip_rows == 2 else list(range(1, self.skip_rows + 1))
            )
            
            # Clean column names
            self.raw_data.columns = self.raw_data.columns.str.strip()
            
            # Verify required columns exist
            if self.iat_column not in self.raw_data.columns:
                raise ValueError(f"Column '{self.iat_column}' not found. Available: {list(self.raw_data.columns)}")
            if self.id_column not in self.raw_data.columns:
                raise ValueError(f"Column '{self.id_column}' not found. Available: {list(self.raw_data.columns)}")
            
            # Filter rows with actual IAT data (must contain 'block' keyword)
            initial_count = len(self.raw_data)
            self.raw_data = self.raw_data[
                self.raw_data[self.iat_column].notna() & 
                self.raw_data[self.iat_column].str.contains('block', na=False, case=False)
            ].copy()
            
            print(f"✅ Found {len(self.raw_data)} participants with SC-IAT data (filtered from {initial_count} rows)")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
            
        return self
    
    def _clean_embedded_csv(self, text: str) -> Optional[str]:
        """
        Clean embedded CSV text from MinnoJS/Qualtrics format.
        
        Handles:
        - Mixed line endings
        - Duplicate header rows
        - Empty lines
        
        NOTE: We do NOT replace "" with " here because pandas read_csv
        handles CSV quote escaping automatically when quoting=csv.QUOTE_ALL
        """
        if pd.isna(text) or text.strip() == '':
            return None
        
        # Step 1: Normalize line endings only
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Step 2: Split into lines and clean
        lines = [line for line in text.split('\n') if line.strip()]
        
        if len(lines) < 2:
            return None
        
        # Step 3: Identify and remove duplicate headers
        header = lines[0]
        data_lines = [lines[0]]  # Keep first header
        for line in lines[1:]:
            if line != header:  # Skip duplicate headers
                data_lines.append(line)
        
        return '\n'.join(data_lines)
    
    def _parse_participant_csv(self, participant_id: str, csv_text: str) -> Optional[pd.DataFrame]:
        """
        Parse embedded CSV for a single participant.
        
        Args:
            participant_id: Participant identifier
            csv_text: Raw embedded CSV text
            
        Returns:
            DataFrame with parsed trials or None if parsing fails
        """
        clean_csv = self._clean_embedded_csv(csv_text)
        
        if clean_csv is None:
            print(f"  ⚠️ Empty/invalid data for participant: {participant_id}")
            return None
        
        try:
            # Parse CSV with proper quote handling
            # doublequote=True means "" is interpreted as an escaped "
            df = pd.read_csv(
                StringIO(clean_csv),
                doublequote=True,
                skipinitialspace=True,
                on_bad_lines='warn'
            )
            
            # Add participant ID
            df['ParticipantID'] = participant_id
            
            # Remove completely empty rows
            df = df.dropna(how='all', subset=[c for c in df.columns if c != 'ParticipantID'])
            
            return df
            
        except Exception as e:
            print(f"  ❌ Parse error for {participant_id}: {e}")
            return None
    
    def _assign_condition(self, block_text: str) -> str:
        """
        Determine trial condition (congruent/incongruent) from block configuration.
        
        SC-IAT condition depends on whether Patient (Paziente) is paired with:
        - Positive attribute (Sostenere/Help) → Congruent
        - Negative attribute (Ignorare/Neglect) → Incongruent
        
        Block text format: "LeftCategories/RightCategories"
        """
        if pd.isna(block_text) or block_text.strip() == '':
            return 'unknown'
        
        block_text = str(block_text).strip()
        
        # Define congruent patterns (Patient + Help on same side)
        # Format: LEFT/RIGHT - categories separated by comma on same side
        congruent_patterns = [
            r'Paziente.*Sostenere',      # Patient with Help on LEFT
            r'Sostenere.*Paziente',      # Help with Patient on LEFT  
            r'Paziente/.*Ignorare',      # Patient LEFT, Neglect RIGHT (Patient alone = needs Help pairing)
            r'Ignorare.*Sostenere/Paziente',  # Attributes LEFT, Patient RIGHT with Help association
        ]
        
        incongruent_patterns = [
            r'Paziente.*Ignorare',       # Patient with Neglect on LEFT
            r'Ignorare.*Paziente',       # Neglect with Patient on LEFT
            r'Sostenere.*Ignorare/Paziente',  # Both attrs LEFT, Patient RIGHT (incongruent)
        ]
        
        # Specific patterns from the actual data
        # Based on R code mappings:
        specific_mappings = {
            'Ignorare,Sostenere/Paziente': 'congruent',
            'Paziente/Ignorare,Sostenere': 'incongruent', 
            'Paziente/Sostenere,Ignorare': 'congruent',
            'Sostenere,Ignorare/Paziente': 'incongruent',
            # Additional variants with different spacing/ordering
            'Paziente,Sostenere/Ignorare': 'congruent',
            'Paziente,Ignorare/Sostenere': 'incongruent',
            'Sostenere/Paziente,Ignorare': 'congruent',
            'Ignorare/Paziente,Sostenere': 'incongruent',
        }
        
        # Clean block text for matching
        clean_text = re.sub(r'\s+', '', block_text)
        
        for pattern, condition in specific_mappings.items():
            clean_pattern = re.sub(r'\s+', '', pattern)
            if clean_text == clean_pattern:
                return condition
        
        # Fallback: try to detect from text structure
        # If Paziente and Sostenere are on same side of '/'
        parts = block_text.split('/')
        if len(parts) == 2:
            left, right = parts
            # Check left side
            if 'Paziente' in left and 'Sostenere' in left:
                return 'congruent'
            if 'Paziente' in left and 'Ignorare' in left:
                return 'incongruent'
            # Check right side
            if 'Paziente' in right and 'Sostenere' in right:
                return 'congruent'
            if 'Paziente' in right and 'Ignorare' in right:
                return 'incongruent'
        
        return 'unknown'
    
    def parse_all_participants(self) -> 'SCIATParser':
        """Parse SC-IAT data for all participants."""
        if self.raw_data is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")
        
        print("\n🔄 Parsing SC-IAT data for all participants...")
        
        parsed_dfs = []
        parse_errors = []
        
        for idx, row in self.raw_data.iterrows():
            participant_id = str(row[self.id_column])
            csv_text = row[self.iat_column]
            
            df = self._parse_participant_csv(participant_id, csv_text)
            
            if df is not None and len(df) > 0:
                parsed_dfs.append(df)
            else:
                parse_errors.append(participant_id)
        
        if len(parsed_dfs) == 0:
            raise RuntimeError("No data successfully parsed!")
        
        # Combine all participants
        self.trial_data = pd.concat(parsed_dfs, ignore_index=True)
        
        print(f"✅ Successfully parsed {len(parsed_dfs)} participants")
        print(f"   Total trials: {len(self.trial_data)}")
        if parse_errors:
            print(f"   ⚠️ Parse errors for {len(parse_errors)} participants: {parse_errors[:5]}{'...' if len(parse_errors) > 5 else ''}")
        
        return self
    
    def process_trials(self) -> 'SCIATParser':
        """Process and enrich trial data with conditions and metadata."""
        if self.trial_data is None:
            raise RuntimeError("No parsed data. Call parse_all_participants() first.")
        
        print("\n🔧 Processing trial data...")
        
        df = self.trial_data.copy()
        
        # Standardize column names (handle variations)
        column_mapping = {
            'block': 'Block',
            'trial': 'Trial', 
            'cond': 'BlockText',
            'type': 'TrialType',
            'cat': 'Category',
            'stim': 'Stimulus',
            'resp': 'Response',
            'err': 'Error',
            'rt': 'RT',
            'd': 'D',
            'fb': 'Feedback',
            'bOrd': 'BlockOrder'
        }
        
        # Rename columns that exist
        for old, new in column_mapping.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
        
        # Convert numeric columns
        for col in ['Block', 'Trial', 'RT', 'Error']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Clean BlockText
        if 'BlockText' in df.columns:
            df['BlockText'] = df['BlockText'].astype(str).str.strip()
            df['BlockText'] = df['BlockText'].str.replace(r'[\r\n]+', '', regex=True)
        
        # Assign trial type/condition
        df['Condition'] = 'unknown'
        
        # Instructions
        if 'TrialType' in df.columns:
            df.loc[df['TrialType'] == 'instructions', 'Condition'] = 'instructions'
        
        # End block
        df.loc[df['Block'] == self.end_block, 'Condition'] = 'end'
        
        # Assign congruent/incongruent for trial blocks
        trial_mask = df['Block'].isin(self.practice_blocks + self.critical_blocks)
        if 'BlockText' in df.columns:
            df.loc[trial_mask, 'Condition'] = df.loc[trial_mask, 'BlockText'].apply(self._assign_condition)
        
        # Mark practice vs critical trials
        df['IsCritical'] = df['Block'].isin(self.critical_blocks)
        df['IsPractice'] = df['Block'].isin(self.practice_blocks)
        
        # Calculate block order per participant (which condition came first)
        block_order = (
            df[df['Condition'].isin(['congruent', 'incongruent'])]
            .sort_values(['ParticipantID', 'Block', 'Trial'])
            .groupby('ParticipantID')['Condition']
            .first()
            .map({'congruent': 0, 'incongruent': 1})
        )
        df['FirstBlockCondition'] = df['ParticipantID'].map(block_order)
        
        # Flag RT outliers (for reference, not exclusion yet)
        df['RT_Timeout'] = df['RT'] >= self.rt_upper_limit
        df['RT_Anticipatory'] = df['RT'] < self.rt_lower_limit
        
        self.trial_data = df
        
        # Print condition distribution
        if 'Condition' in df.columns:
            print("\n   Condition distribution:")
            cond_counts = df['Condition'].value_counts()
            for cond, count in cond_counts.items():
                print(f"     {cond}: {count}")
        
        return self
    
    def compute_quality_metrics(self) -> 'SCIATParser':
        """Compute data quality metrics for exclusion decisions."""
        if self.trial_data is None:
            raise RuntimeError("No processed data. Call process_trials() first.")
        
        print("\n📊 Computing quality metrics...")
        
        df = self.trial_data
        
        quality_data = []
        
        for pid in df['ParticipantID'].unique():
            p_all = df[df['ParticipantID'] == pid]
            
            # Filter out instruction trials (stim0) for all calculations
            p_trials = p_all[
                (p_all['TrialType'] != 'instructions') &
                (p_all['Category'] != 'stim0')
            ]
            
            # Separate by block type
            p_practice = p_trials[p_trials['IsPractice'] == True]
            p_critical = p_trials[p_trials['IsCritical'] == True]
            
            # Helper function to compute stats for a set of trials
            def compute_trial_stats(trials_df, prefix):
                n_total = len(trials_df)
                
                if n_total == 0:
                    return {
                        f'{prefix}_N_Total': 0,
                        f'{prefix}_N_Correct': 0,
                        f'{prefix}_N_Errors': 0,
                        f'{prefix}_N_Timeouts': 0,
                        f'{prefix}_N_Anticipatory': 0,
                        f'{prefix}_N_Valid': 0,
                        f'{prefix}_Error_Rate': np.nan,
                        f'{prefix}_Timeout_Rate': np.nan,
                        f'{prefix}_Mean_RT': np.nan,
                        f'{prefix}_SD_RT': np.nan,
                    }
                
                # MinnoJS error coding: 0=error, 1=correct, 2=timeout
                n_correct = (trials_df['Error'] == 1).sum()
                n_errors = (trials_df['Error'] == 0).sum()
                n_timeout = (trials_df['Error'] == 2).sum()
                
                # Anticipatory responses (RT < 350ms, only for non-timeout trials)
                non_timeout = trials_df[trials_df['Error'] != 2]
                n_anticipatory = (non_timeout['RT'] < self.rt_lower_limit).sum() if len(non_timeout) > 0 else 0
                
                # Valid trials = total - timeouts - anticipatory
                n_valid = n_total - n_timeout - n_anticipatory
                
                # Error rate: errors / (correct + errors), excluding timeouts
                n_responses = n_correct + n_errors
                error_rate = n_errors / n_responses if n_responses > 0 else np.nan
                
                # Timeout rate
                timeout_rate = n_timeout / n_total if n_total > 0 else np.nan
                
                # RT stats (excluding timeouts and anticipatory)
                valid_rt = non_timeout[non_timeout['RT'] >= self.rt_lower_limit]['RT']
                mean_rt = valid_rt.mean() if len(valid_rt) > 0 else np.nan
                sd_rt = valid_rt.std() if len(valid_rt) > 0 else np.nan
                
                return {
                    f'{prefix}_N_Total': n_total,
                    f'{prefix}_N_Correct': n_correct,
                    f'{prefix}_N_Errors': n_errors,
                    f'{prefix}_N_Timeouts': n_timeout,
                    f'{prefix}_N_Anticipatory': n_anticipatory,
                    f'{prefix}_N_Valid': n_valid,
                    f'{prefix}_Error_Rate': round(error_rate, 4) if not np.isnan(error_rate) else np.nan,
                    f'{prefix}_Timeout_Rate': round(timeout_rate, 4) if not np.isnan(timeout_rate) else np.nan,
                    f'{prefix}_Mean_RT': round(mean_rt, 2) if not np.isnan(mean_rt) else np.nan,
                    f'{prefix}_SD_RT': round(sd_rt, 2) if not np.isnan(sd_rt) else np.nan,
                }
            
            # Compute stats for each set
            practice_stats = compute_trial_stats(p_practice, 'Practice')
            critical_stats = compute_trial_stats(p_critical, 'Critical')
            all_stats = compute_trial_stats(p_trials, 'All')
            
            # Block completeness
            blocks_present = p_all['Block'].unique()
            has_block_2 = 2 in blocks_present
            has_block_4 = 4 in blocks_present
            
            # Condition balance (critical only)
            n_congruent = len(p_critical[p_critical['Condition'] == 'congruent'])
            n_incongruent = len(p_critical[p_critical['Condition'] == 'incongruent'])
            
            # First block condition
            first_condition = p_all[p_all['Condition'].isin(['congruent', 'incongruent'])].sort_values(['Block', 'Trial'])
            first_cond = first_condition['Condition'].iloc[0] if len(first_condition) > 0 else 'unknown'
            
            # Exclusion flags based on CRITICAL trials (the 48 trials)
            critical_error_rate = critical_stats['Critical_Error_Rate']
            critical_timeout_rate = critical_stats['Critical_Timeout_Rate']
            critical_valid = critical_stats['Critical_N_Valid']
            
            # Two thresholds for error rate: 20% (strict) and 40% (standard)
            exclude_error_rate_20 = critical_error_rate > 0.20 if not np.isnan(critical_error_rate) else True
            exclude_error_rate_40 = critical_error_rate > 0.40 if not np.isnan(critical_error_rate) else True
            exclude_timeout_rate = critical_timeout_rate >= self.max_timeout_rate if not np.isnan(critical_timeout_rate) else True
            exclude_few_trials = critical_valid < self.min_valid_trials
            exclude_missing_blocks = not (has_block_2 and has_block_4)
            
            # Combined exclusion flags
            exclude_strict = exclude_error_rate_20 or exclude_timeout_rate or exclude_few_trials or exclude_missing_blocks
            exclude_standard = exclude_error_rate_40 or exclude_timeout_rate or exclude_few_trials or exclude_missing_blocks
            
            # Build row
            row = {'ParticipantID': pid}
            row.update(practice_stats)
            row.update(critical_stats)
            row.update(all_stats)
            row.update({
                'Has_Block_2': has_block_2,
                'Has_Block_4': has_block_4,
                'N_Congruent': n_congruent,
                'N_Incongruent': n_incongruent,
                'First_Condition': first_cond,
                'Exclude_ErrorRate_20': exclude_error_rate_20,
                'Exclude_ErrorRate_40': exclude_error_rate_40,
                'Exclude_TimeoutRate': exclude_timeout_rate,
                'Exclude_FewTrials': exclude_few_trials,
                'Exclude_MissingBlocks': exclude_missing_blocks,
                'EXCLUDE_Strict_20': exclude_strict,
                'EXCLUDE_Standard_40': exclude_standard,
            })
            
            quality_data.append(row)
        
        self.quality_report = pd.DataFrame(quality_data)
        
        # Print summary
        n_total = len(self.quality_report)
        n_exclude_strict = self.quality_report['EXCLUDE_Strict_20'].sum()
        n_exclude_standard = self.quality_report['EXCLUDE_Standard_40'].sum()
        
        print(f"\n   Quality summary:")
        print(f"     Total participants: {n_total}")
        print(f"     Exclusions with 20% error threshold (strict): {n_exclude_strict} ({100*n_exclude_strict/n_total:.1f}%)")
        print(f"     Exclusions with 40% error threshold (standard): {n_exclude_standard} ({100*n_exclude_standard/n_total:.1f}%)")
        print(f"       - Error rate >20%: {self.quality_report['Exclude_ErrorRate_20'].sum()}")
        print(f"       - Error rate >40%: {self.quality_report['Exclude_ErrorRate_40'].sum()}")
        print(f"       - Timeout rate ≥8.33%: {self.quality_report['Exclude_TimeoutRate'].sum()}")
        print(f"       - Too few valid trials (<{self.min_valid_trials}): {self.quality_report['Exclude_FewTrials'].sum()}")
        print(f"       - Missing blocks: {self.quality_report['Exclude_MissingBlocks'].sum()}")
        
        return self
    
    def compute_summary_statistics(self) -> 'SCIATParser':
        """Compute per-participant summary statistics for D-score calculation."""
        if self.trial_data is None:
            raise RuntimeError("No processed data. Call process_trials() first.")
        
        print("\n📈 Computing summary statistics...")
        
        df = self.trial_data
        
        # Filter critical trials only, exclude timeouts (Error == 2) and instruction trials
        critical = df[
            (df['IsCritical'] == True) & 
            (df['Error'] != 2) &
            (df['TrialType'] != 'instructions') &
            (df['Category'] != 'stim0')
        ].copy()
        
        # Custom aggregation function to count error codes correctly
        def count_correct(x):
            return (x == 1).sum()
        
        def count_errors(x):
            return (x == 0).sum()
        
        def error_rate(x):
            n_correct = (x == 1).sum()
            n_errors = (x == 0).sum()
            total = n_correct + n_errors
            return n_errors / total if total > 0 else np.nan
        
        # Group by participant and condition
        summary = critical.groupby(['ParticipantID', 'Condition']).agg({
            'Trial': 'count',
            'Error': [count_correct, count_errors, error_rate],
            'RT': ['mean', 'std', 'median']
        }).reset_index()
        
        # Flatten column names
        summary.columns = [
            'ParticipantID', 'Condition',
            'N_Trials', 'N_Correct', 'N_Errors', 'Error_Rate',
            'Mean_RT', 'SD_RT', 'Median_RT'
        ]
        
        # Pivot to wide format
        summary_wide = summary.pivot(
            index='ParticipantID',
            columns='Condition',
            values=['N_Trials', 'N_Correct', 'N_Errors', 'Error_Rate', 'Mean_RT', 'SD_RT', 'Median_RT']
        )
        
        # Flatten column names
        summary_wide.columns = ['_'.join(col).strip() for col in summary_wide.columns.values]
        summary_wide = summary_wide.reset_index()
        
        # Add first block condition
        first_cond = (
            df[df['Condition'].isin(['congruent', 'incongruent'])]
            .sort_values(['ParticipantID', 'Block', 'Trial'])
            .groupby('ParticipantID')['Condition']
            .first()
        )
        summary_wide['First_Condition'] = summary_wide['ParticipantID'].map(first_cond)
        
        # Calculate pooled SD and D-score if both conditions exist
        if 'Mean_RT_congruent' in summary_wide.columns and 'Mean_RT_incongruent' in summary_wide.columns:
            # Pooled SD (using the Karpinski & Steinman formula)
            sd_cong = summary_wide.get('SD_RT_congruent', np.nan)
            sd_incong = summary_wide.get('SD_RT_incongruent', np.nan)
            
            summary_wide['SD_Pooled'] = np.sqrt((sd_cong**2 + sd_incong**2) / 2)
            
            # D-score: (M_incongruent - M_congruent) / SD_pooled
            # Positive D = faster for congruent (patient + help) = implicit positive association
            summary_wide['D_Score'] = (
                (summary_wide['Mean_RT_incongruent'] - summary_wide['Mean_RT_congruent']) / 
                summary_wide['SD_Pooled']
            )
        
        self.summary_data = summary_wide
        
        print(f"✅ Summary computed for {len(summary_wide)} participants")
        
        return self
    
    def export(self, output_prefix: str = 'SCIAT_parsed') -> 'SCIATParser':
        """Export all processed data to CSV files."""
        print(f"\n💾 Exporting data with prefix: {output_prefix}")
        
        output_dir = Path(output_prefix).parent if '/' in output_prefix else Path('.')
        output_dir.mkdir(exist_ok=True)
        
        # Trial-level data
        if self.trial_data is not None:
            trial_file = f"{output_prefix}_trial_level.csv"
            self.trial_data.to_csv(trial_file, index=False)
            print(f"   ✓ Trial-level data: {trial_file}")
        
        # Quality report
        if self.quality_report is not None:
            quality_file = f"{output_prefix}_quality_report.csv"
            self.quality_report.to_csv(quality_file, index=False)
            print(f"   ✓ Quality report: {quality_file}")
        
        # Summary statistics
        if self.summary_data is not None:
            summary_file = f"{output_prefix}_summary.csv"
            self.summary_data.to_csv(summary_file, index=False)
            print(f"   ✓ Summary statistics: {summary_file}")
        
        # Combined Excel workbook
        try:
            excel_file = f"{output_prefix}_complete.xlsx"
            with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                if self.trial_data is not None:
                    self.trial_data.to_excel(writer, sheet_name='Trial_Level', index=False)
                if self.quality_report is not None:
                    self.quality_report.to_excel(writer, sheet_name='Quality_Report', index=False)
                if self.summary_data is not None:
                    self.summary_data.to_excel(writer, sheet_name='Summary_Stats', index=False)
            print(f"   ✓ Excel workbook: {excel_file}")
        except ImportError:
            print("   ⚠️ openpyxl not available, skipping Excel export")
        
        return self
    
    def run(self, output_prefix: str = 'SCIAT_parsed') -> 'SCIATParser':
        """Run the complete parsing pipeline."""
        self.load_data()
        self.parse_all_participants()
        self.process_trials()
        self.compute_quality_metrics()
        self.compute_summary_statistics()
        self.export(output_prefix)
        
        print("\n" + "=" * 60)
        print("🎉 SC-IAT parsing completed successfully!")
        print("=" * 60)
        
        return self


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Parse SC-IAT data from Qualtrics/MinnoJS exports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python sciat_minnojs_parser.py RAW_imm.csv
    python sciat_minnojs_parser.py RAW_imm.csv output/study3 --iat-column SC_IAT_aiuto
    python sciat_minnojs_parser.py data.csv results --id-column SubjectID
        """
    )
    
    parser.add_argument('input_file', help='Input Qualtrics CSV file')
    parser.add_argument('output_prefix', nargs='?', default='SCIAT_parsed',
                        help='Output file prefix (default: SCIAT_parsed)')
    parser.add_argument('--iat-column', default='SC_IAT_aiuto',
                        help='Column name containing SC-IAT data (default: SC_IAT_aiuto)')
    parser.add_argument('--id-column', default='ResponseId',
                        help='Column name containing participant IDs (default: ResponseId)')
    parser.add_argument('--skip-rows', type=int, default=2,
                        help='Number of Qualtrics metadata rows to skip (default: 2)')
    
    args = parser.parse_args()
    
    # Run parser
    sciat = SCIATParser(
        args.input_file,
        iat_column=args.iat_column,
        id_column=args.id_column,
        skip_rows=args.skip_rows
    )
    sciat.run(args.output_prefix)


if __name__ == '__main__':
    main()