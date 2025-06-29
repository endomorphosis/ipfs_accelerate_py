#!/usr/bin/env python3
"""
Transforms Module for Result Aggregator Pipeline

This module provides a collection of transformation classes that can be used
in processing pipelines for test results data. Each transform takes a DataFrame
as input, applies a specific transformation, and returns a transformed DataFrame.

Usage:
    from result_aggregator.pipeline.transforms import FilterTransform, AggregateTransform
    
    # Create filter transform to select specific test results
    filter_transform = FilterTransform(model_name='bert-base-uncased', hardware_type='cuda')
    
    # Create aggregation transform to calculate statistics
    aggregate_transform = AggregateTransform(
        group_by='hardware_type', 
        metrics=['latency', 'throughput']
    )
    
    # Apply transforms to data
    filtered_data = filter_transform.transform(data)
    aggregated_data = aggregate_transform.transform(filtered_data)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import re
import logging
from abc import ABC, abstractmethod
from result_aggregator.pipeline.pipeline import Transform

logger = logging.getLogger(__name__)


class FilterTransform(Transform):
    """
    Filter data based on specific criteria.
    
    This transform filters rows in a DataFrame based on exact match or
    pattern match conditions for specified columns.
    """
    
    def __init__(self, 
                **filter_criteria: Any):
        """
        Initialize the filter transform.
        
        Args:
            **filter_criteria: Column-value pairs for filtering
                Values can be:
                - Scalar values for exact match
                - Lists for "in" condition
                - Tuples with (operator, value) for comparison
                - Regular expression pattern strings for pattern matching
        """
        self.filter_criteria = filter_criteria
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply filtering to the input data.
        
        Args:
            data: Input DataFrame
        
        Returns:
            Filtered DataFrame
        """
        if not self.filter_criteria:
            return data
        
        filtered_data = data.copy()
        initial_count = len(filtered_data)
        
        for column, value in self.filter_criteria.items():
            if column not in filtered_data.columns:
                logger.warning(f"Column '{column}' not found in data, skipping filter")
                continue
            
            if isinstance(value, (list, set, tuple)) and not isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
                # Handle comparison operators
                filtered_data = self._apply_comparison_filter(filtered_data, column, value)
            elif isinstance(value, (list, set)):
                # Handle "in" condition
                filtered_data = filtered_data[filtered_data[column].isin(value)]
            elif isinstance(value, str) and value.startswith('regex:'):
                # Handle regex pattern
                pattern = value.replace('regex:', '', 1)
                filtered_data = filtered_data[filtered_data[column].astype(str).str.match(pattern)]
            elif callable(value):
                # Handle callable predicate
                filtered_data = filtered_data[filtered_data[column].apply(value)]
            else:
                # Handle exact match
                filtered_data = filtered_data[filtered_data[column] == value]
        
        filtered_count = len(filtered_data)
        logger.info(f"Filtered data from {initial_count} to {filtered_count} rows")
        
        return filtered_data
    
    def _apply_comparison_filter(self, data: pd.DataFrame, column: str, filter_value: Tuple) -> pd.DataFrame:
        """
        Apply a comparison filter using operators like >, <, >=, <=, !=.
        
        Args:
            data: Input DataFrame
            column: Column to filter on
            filter_value: Tuple of (operator, value) for comparison
        
        Returns:
            Filtered DataFrame
        """
        operator, value = filter_value
        
        if operator == '>':
            return data[data[column] > value]
        elif operator == '>=':
            return data[data[column] >= value]
        elif operator == '<':
            return data[data[column] < value]
        elif operator == '<=':
            return data[data[column] <= value]
        elif operator == '!=':
            return data[data[column] != value]
        else:
            logger.warning(f"Unsupported operator: {operator}, skipping filter")
            return data


class TimeWindowTransform(Transform):
    """
    Filter data based on a time window.
    
    This transform filters rows in a DataFrame to include only those
    within a specified time window.
    """
    
    def __init__(self, 
                days: Optional[int] = None,
                hours: Optional[int] = None,
                start_date: Optional[Union[str, datetime]] = None,
                end_date: Optional[Union[str, datetime]] = None,
                timestamp_column: str = 'timestamp'):
        """
        Initialize the time window transform.
        
        Args:
            days: Number of days to look back from now
            hours: Number of hours to look back from now
            start_date: Start date for the time window
            end_date: End date for the time window
            timestamp_column: Name of the timestamp column
        
        Raises:
            ValueError: If neither days/hours nor start_date is provided
        """
        self.days = days
        self.hours = hours
        
        if isinstance(start_date, str):
            self.start_date = pd.to_datetime(start_date)
        else:
            self.start_date = start_date
            
        if isinstance(end_date, str):
            self.end_date = pd.to_datetime(end_date)
        else:
            self.end_date = end_date
            
        self.timestamp_column = timestamp_column
        
        if not any([days is not None, hours is not None, start_date is not None]):
            raise ValueError("Either days/hours or start_date must be provided")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply time window filtering to the input data.
        
        Args:
            data: Input DataFrame
        
        Returns:
            Time-filtered DataFrame
        """
        if self.timestamp_column not in data.columns:
            logger.warning(f"Column '{self.timestamp_column}' not found in data, returning original data")
            return data
        
        # Convert timestamp column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data[self.timestamp_column]):
            data = data.copy()
            data[self.timestamp_column] = pd.to_datetime(data[self.timestamp_column])
        
        initial_count = len(data)
        
        # Calculate the start date if days/hours are provided
        if self.days is not None or self.hours is not None:
            now = datetime.now()
            days = self.days or 0
            hours = self.hours or 0
            start_date = now - timedelta(days=days, hours=hours)
        else:
            start_date = self.start_date
        
        # Apply start date filter
        if start_date is not None:
            data = data[data[self.timestamp_column] >= start_date]
        
        # Apply end date filter
        if self.end_date is not None:
            data = data[data[self.timestamp_column] <= self.end_date]
        
        filtered_count = len(data)
        
        time_range = f"from {start_date} to {self.end_date or 'now'}"
        logger.info(f"Time window filter {time_range} reduced data from {initial_count} to {filtered_count} rows")
        
        return data


class AggregateTransform(Transform):
    """
    Aggregate data based on grouping columns and metrics.
    
    This transform performs grouping and aggregation on the input data,
    calculating statistics for specified metrics.
    """
    
    def __init__(self, 
                group_by: Union[str, List[str]],
                metrics: List[str],
                aggregations: Optional[Dict[str, Union[str, List[str]]]] = None,
                pivot: bool = False,
                pivot_column: Optional[str] = None,
                pivot_values: Optional[str] = None):
        """
        Initialize the aggregate transform.
        
        Args:
            group_by: Column(s) to group by
            metrics: List of metric columns to aggregate
            aggregations: Optional dictionary mapping metrics to aggregation functions
                Default aggregations: mean, min, max, std, count
            pivot: Whether to pivot the result
            pivot_column: Column to use for pivot (required if pivot=True)
            pivot_values: Values column for pivot (required if pivot=True)
        
        Raises:
            ValueError: If invalid parameters are provided
        """
        self.group_by = [group_by] if isinstance(group_by, str) else group_by
        self.metrics = metrics
        
        # Default aggregations for each metric
        default_aggs = ['mean', 'min', 'max', 'std', 'count']
        
        if aggregations is None:
            self.aggregations = {metric: default_aggs for metric in metrics}
        else:
            self.aggregations = aggregations
        
        self.pivot = pivot
        if pivot:
            if not pivot_column:
                raise ValueError("pivot_column must be provided if pivot=True")
            if not pivot_values:
                raise ValueError("pivot_values must be provided if pivot=True")
        
        self.pivot_column = pivot_column
        self.pivot_values = pivot_values
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply aggregation to the input data.
        
        Args:
            data: Input DataFrame
        
        Returns:
            Aggregated DataFrame
        """
        # Check if all required columns exist
        missing_columns = []
        
        for col in self.group_by:
            if col not in data.columns:
                missing_columns.append(col)
        
        for col in self.metrics:
            if col not in data.columns:
                missing_columns.append(col)
        
        if missing_columns:
            logger.warning(f"Columns not found in data: {', '.join(missing_columns)}")
            
            # Filter out missing metrics
            valid_metrics = [m for m in self.metrics if m in data.columns]
            if not valid_metrics:
                logger.error("No valid metrics found for aggregation")
                return pd.DataFrame()
            
            self.metrics = valid_metrics
            
            # Filter out missing group by columns
            valid_group_by = [g for g in self.group_by if g in data.columns]
            if not valid_group_by:
                logger.error("No valid columns for grouping")
                return pd.DataFrame()
            
            self.group_by = valid_group_by
        
        # Prepare a pivot if requested
        if self.pivot:
            if self.pivot_column not in data.columns:
                logger.error(f"Pivot column '{self.pivot_column}' not found in data")
                return pd.DataFrame()
            
            if self.pivot_values not in data.columns:
                logger.error(f"Pivot values column '{self.pivot_values}' not found in data")
                return pd.DataFrame()
            
            # Pivot the data
            pivot_table = pd.pivot_table(
                data,
                values=self.pivot_values,
                index=self.group_by,
                columns=self.pivot_column,
                aggfunc='mean'  # Default aggregation for pivot
            )
            
            return pivot_table.reset_index()
        
        # For standard aggregation, filter to relevant columns only
        relevant_columns = self.group_by + self.metrics
        filtered_data = data[relevant_columns].copy()
        
        # Create a list of aggregation operations to apply
        agg_dict = {}
        for metric in self.metrics:
            if metric in self.aggregations:
                agg_dict[metric] = self.aggregations[metric]
            else:
                agg_dict[metric] = ['mean', 'min', 'max', 'std', 'count']
        
        # Perform the aggregation
        grouped = filtered_data.groupby(self.group_by)
        aggregated = grouped.agg(agg_dict)
        
        # Flatten multi-level column index
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
        
        # Reset index to convert group by columns back to regular columns
        result = aggregated.reset_index()
        
        logger.info(f"Aggregated data: {len(data)} rows → {len(result)} groups")
        
        return result


class PivotTransform(Transform):
    """
    Pivot data to create a cross-tabulation.
    
    This transform creates a pivot table from the input data,
    with rows from one column and columns from another.
    """
    
    def __init__(self, 
                index: Union[str, List[str]],
                columns: str,
                values: str,
                aggfunc: str = 'mean',
                fill_value: Optional[Any] = None):
        """
        Initialize the pivot transform.
        
        Args:
            index: Column(s) to use as index (row labels)
            columns: Column to use for column labels
            values: Column to use for values
            aggfunc: Aggregation function to use ('mean', 'sum', 'count', etc.)
            fill_value: Value to use for missing data
        """
        self.index = index
        self.columns = columns
        self.values = values
        self.aggfunc = aggfunc
        self.fill_value = fill_value
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply pivot transformation to the input data.
        
        Args:
            data: Input DataFrame
        
        Returns:
            Pivoted DataFrame
        """
        # Check if all required columns exist
        required_columns = [self.columns, self.values]
        if isinstance(self.index, str):
            required_columns.append(self.index)
        else:
            required_columns.extend(self.index)
        
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logger.error(f"Columns not found in data: {', '.join(missing_columns)}")
            return pd.DataFrame()
        
        # Create the pivot table
        pivot_table = pd.pivot_table(
            data,
            values=self.values,
            index=self.index,
            columns=self.columns,
            aggfunc=self.aggfunc,
            fill_value=self.fill_value
        )
        
        # Reset index to convert index columns back to regular columns
        result = pivot_table.reset_index()
        
        logger.info(f"Pivoted data: {len(data)} rows → {len(result)} rows with {len(pivot_table.columns)} columns")
        
        return result


class MetricPivotTransform(Transform):
    """
    Pivot metrics from rows to columns.
    
    This transform is specifically designed for test results data where
    metrics are stored in a 'metric_name'/'metric_value' format and need
    to be pivoted to columns.
    """
    
    def __init__(self,
                metric_name_column: str = 'metric_name',
                metric_value_column: str = 'metric_value',
                group_by: Union[str, List[str]] = None):
        """
        Initialize the metric pivot transform.
        
        Args:
            metric_name_column: Column containing metric names
            metric_value_column: Column containing metric values
            group_by: Optional column(s) to group by before pivoting
        """
        self.metric_name_column = metric_name_column
        self.metric_value_column = metric_value_column
        
        if group_by is None:
            # Default group by is all columns except metric name and value
            self.group_by = None
        else:
            self.group_by = [group_by] if isinstance(group_by, str) else group_by
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply metric pivoting to the input data.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with metrics as columns
        """
        # Check if required columns exist
        if self.metric_name_column not in data.columns:
            logger.error(f"Metric name column '{self.metric_name_column}' not found in data")
            return data
        
        if self.metric_value_column not in data.columns:
            logger.error(f"Metric value column '{self.metric_value_column}' not found in data")
            return data
        
        # Determine group by columns if not explicitly provided
        if self.group_by is None:
            self.group_by = [col for col in data.columns if col not in 
                          [self.metric_name_column, self.metric_value_column]]
        
        # Check if all group by columns exist
        missing_columns = [col for col in self.group_by if col not in data.columns]
        if missing_columns:
            logger.warning(f"Group by columns not found: {', '.join(missing_columns)}")
            # Use only existing columns
            self.group_by = [col for col in self.group_by if col in data.columns]
            
            if not self.group_by:
                logger.error("No valid columns for grouping")
                return data
        
        # Create a pivot table
        try:
            pivot_table = data.pivot_table(
                index=self.group_by,
                columns=self.metric_name_column,
                values=self.metric_value_column,
                aggfunc='first'  # Take first value when duplicates exist
            )
            
            # Convert multi-index to regular columns
            result = pivot_table.reset_index()
            
            logger.info(f"Pivoted metrics: {data[self.metric_name_column].nunique()} unique metrics as columns")
            
            return result
            
        except Exception as e:
            logger.error(f"Error pivoting metrics: {str(e)}")
            return data


class CalculatedMetricTransform(Transform):
    """
    Add calculated metrics based on existing metrics.
    
    This transform adds new columns to the DataFrame with values calculated
    from existing metrics using provided formulas.
    """
    
    def __init__(self, **metric_formulas: Callable):
        """
        Initialize the calculated metric transform.
        
        Args:
            **metric_formulas: Mapping of new metric names to calculator functions
                Each function should take a pandas DataFrame row as input and return the calculated value
        """
        self.metric_formulas = metric_formulas
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply calculated metric transformations to the input data.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with additional calculated metrics
        """
        if not self.metric_formulas:
            return data
        
        result = data.copy()
        
        for metric_name, formula in self.metric_formulas.items():
            try:
                result[metric_name] = result.apply(formula, axis=1)
                logger.info(f"Added calculated metric: {metric_name}")
            except Exception as e:
                logger.error(f"Error calculating metric '{metric_name}': {str(e)}")
        
        return result


class ZScoreOutlierTransform(Transform):
    """
    Detect and optionally remove outliers using Z-scores.
    
    This transform identifies outliers in specified columns using Z-scores,
    and can either remove them or add an outlier flag.
    """
    
    def __init__(self,
                columns: Union[str, List[str]],
                threshold: float = 3.0,
                remove_outliers: bool = False,
                add_outlier_flag: bool = True,
                flag_column: str = 'is_outlier'):
        """
        Initialize the Z-score outlier transform.
        
        Args:
            columns: Column(s) to check for outliers
            threshold: Z-score threshold for outlier detection
            remove_outliers: Whether to remove outliers from the result
            add_outlier_flag: Whether to add a flag column for outliers
            flag_column: Name of the outlier flag column
        """
        self.columns = [columns] if isinstance(columns, str) else columns
        self.threshold = threshold
        self.remove_outliers = remove_outliers
        self.add_outlier_flag = add_outlier_flag
        self.flag_column = flag_column
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply outlier detection to the input data.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with outliers processed as specified
        """
        # Check if columns exist
        valid_columns = [col for col in self.columns if col in data.columns]
        
        if not valid_columns:
            logger.warning(f"None of the specified columns {self.columns} exist in the data")
            return data
        
        # Create output dataframe
        result = data.copy()
        
        # Calculate z-scores for each column
        z_scores = pd.DataFrame(index=data.index)
        
        for col in valid_columns:
            col_mean = data[col].mean()
            col_std = data[col].std()
            
            if col_std == 0 or pd.isna(col_std):
                logger.warning(f"Column '{col}' has zero or NaN standard deviation, skipping")
                continue
            
            z_scores[col] = (data[col] - col_mean) / col_std
        
        # Identify outliers (any column with abs(z) > threshold)
        is_outlier = pd.Series(False, index=data.index)
        
        for col in z_scores.columns:
            is_outlier |= (z_scores[col].abs() > self.threshold)
        
        # Add outlier flag if requested
        if self.add_outlier_flag:
            result[self.flag_column] = is_outlier
        
        # Remove outliers if requested
        if self.remove_outliers:
            result = result[~is_outlier]
            
            outlier_count = sum(is_outlier)
            logger.info(f"Removed {outlier_count} outliers ({outlier_count/len(data):.1%} of data)")
        
        return result


class NormalizationTransform(Transform):
    """
    Normalize numerical columns to a standard scale.
    
    This transform applies normalization techniques to specified columns,
    such as min-max scaling, z-score normalization, or custom range scaling.
    """
    
    def __init__(self,
                columns: Union[str, List[str]],
                method: str = 'min-max',
                custom_range: Optional[Tuple[float, float]] = None):
        """
        Initialize the normalization transform.
        
        Args:
            columns: Column(s) to normalize
            method: Normalization method ('min-max', 'z-score', 'custom')
            custom_range: Target range for custom normalization (min, max)
        
        Raises:
            ValueError: If invalid method or parameters are provided
        """
        self.columns = [columns] if isinstance(columns, str) else columns
        self.method = method.lower()
        self.custom_range = custom_range
        
        valid_methods = ['min-max', 'z-score', 'custom']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid normalization method: {method}. Valid methods: {', '.join(valid_methods)}")
        
        if self.method == 'custom' and custom_range is None:
            raise ValueError("custom_range must be provided for custom normalization")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply normalization to the input data.
        
        Args:
            data: Input DataFrame
        
        Returns:
            DataFrame with normalized columns
        """
        # Check if columns exist
        valid_columns = [col for col in self.columns if col in data.columns]
        
        if not valid_columns:
            logger.warning(f"None of the specified columns {self.columns} exist in the data")
            return data
        
        # Create output dataframe
        result = data.copy()
        
        for col in valid_columns:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(data[col]):
                logger.warning(f"Column '{col}' is not numeric, skipping normalization")
                continue
            
            if self.method == 'min-max':
                # Min-max normalization to [0, 1]
                min_val = data[col].min()
                max_val = data[col].max()
                
                if max_val == min_val:
                    logger.warning(f"Column '{col}' has equal min and max values, setting to constant")
                    result[col] = 0.5  # Middle of range
                else:
                    result[col] = (data[col] - min_val) / (max_val - min_val)
                
            elif self.method == 'z-score':
                # Z-score normalization
                mean = data[col].mean()
                std = data[col].std()
                
                if std == 0 or pd.isna(std):
                    logger.warning(f"Column '{col}' has zero or NaN standard deviation, setting to constant")
                    result[col] = 0  # Z-score of mean
                else:
                    result[col] = (data[col] - mean) / std
                
            elif self.method == 'custom':
                # Custom range normalization
                min_val = data[col].min()
                max_val = data[col].max()
                target_min, target_max = self.custom_range
                
                if max_val == min_val:
                    logger.warning(f"Column '{col}' has equal min and max values, setting to middle of target range")
                    result[col] = (target_min + target_max) / 2
                else:
                    normalized = (data[col] - min_val) / (max_val - min_val)
                    result[col] = normalized * (target_max - target_min) + target_min
        
        return result


class MergeTransform(Transform):
    """
    Merge the input DataFrame with another DataFrame.
    
    This transform performs a database-style join operation between
    the input DataFrame and another DataFrame from a data source.
    """
    
    def __init__(self,
                right_data: Union[pd.DataFrame, str],
                how: str = 'inner',
                left_on: Union[str, List[str]] = None,
                right_on: Union[str, List[str]] = None,
                suffixes: Tuple[str, str] = ('_x', '_y')):
        """
        Initialize the merge transform.
        
        Args:
            right_data: Right DataFrame or path to a data file
            how: Type of join ('inner', 'left', 'right', 'outer')
            left_on: Column(s) from the left DataFrame to join on
            right_on: Column(s) from the right DataFrame to join on
            suffixes: Suffixes to add to overlapping column names
        
        Raises:
            ValueError: If invalid parameters are provided
        """
        self.right_data = right_data
        self.how = how
        self.left_on = [left_on] if isinstance(left_on, str) else left_on
        self.right_on = [right_on] if isinstance(right_on, str) else right_on
        self.suffixes = suffixes
        
        valid_join_types = ['inner', 'left', 'right', 'outer']
        if self.how not in valid_join_types:
            raise ValueError(f"Invalid join type: {how}. Valid types: {', '.join(valid_join_types)}")
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply merge transformation to the input data.
        
        Args:
            data: Input DataFrame
        
        Returns:
            Merged DataFrame
        """
        # Get the right DataFrame
        if isinstance(self.right_data, pd.DataFrame):
            right_df = self.right_data
        elif isinstance(self.right_data, str):
            # Assume it's a file path
            try:
                file_ext = os.path.splitext(self.right_data)[1].lower()
                
                if file_ext == '.csv':
                    right_df = pd.read_csv(self.right_data)
                elif file_ext in ['.json', '.jsonl']:
                    right_df = pd.read_json(self.right_data)
                elif file_ext == '.parquet':
                    right_df = pd.read_parquet(self.right_data)
                elif file_ext in ['.xlsx', '.xls']:
                    right_df = pd.read_excel(self.right_data)
                else:
                    logger.error(f"Unsupported file format: {file_ext}")
                    return data
            except Exception as e:
                logger.error(f"Error reading right DataFrame: {str(e)}")
                return data
        else:
            logger.error(f"Invalid right_data type: {type(self.right_data)}")
            return data
        
        # Check if join columns exist
        if self.left_on:
            missing_left = [col for col in self.left_on if col not in data.columns]
            if missing_left:
                logger.error(f"Left join columns not found: {', '.join(missing_left)}")
                return data
        
        if self.right_on:
            missing_right = [col for col in self.right_on if col not in right_df.columns]
            if missing_right:
                logger.error(f"Right join columns not found: {', '.join(missing_right)}")
                return data
        
        # Perform the merge
        try:
            result = data.merge(
                right_df,
                how=self.how,
                left_on=self.left_on,
                right_on=self.right_on,
                suffixes=self.suffixes
            )
            
            logger.info(f"Merged with {len(right_df)} rows, result has {len(result)} rows")
            
            return result
        except Exception as e:
            logger.error(f"Error performing merge: {str(e)}")
            return data


class SortTransform(Transform):
    """
    Sort the DataFrame by specified columns.
    
    This transform sorts the input DataFrame based on one or more columns
    and specified sort order.
    """
    
    def __init__(self,
                by: Union[str, List[str]],
                ascending: Union[bool, List[bool]] = True):
        """
        Initialize the sort transform.
        
        Args:
            by: Column(s) to sort by
            ascending: Whether to sort in ascending order
                Can be a single boolean or a list of booleans (one per column)
        """
        self.by = [by] if isinstance(by, str) else by
        
        if isinstance(ascending, bool):
            self.ascending = [ascending] * len(self.by)
        else:
            self.ascending = ascending
            
            # Ensure ascending list matches the length of by list
            if len(self.ascending) != len(self.by):
                logger.warning(f"Length mismatch between 'by' ({len(self.by)}) and 'ascending' ({len(self.ascending)})")
                # Extend or truncate the ascending list
                if len(self.ascending) < len(self.by):
                    self.ascending.extend([True] * (len(self.by) - len(self.ascending)))
                else:
                    self.ascending = self.ascending[:len(self.by)]
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply sorting to the input data.
        
        Args:
            data: Input DataFrame
        
        Returns:
            Sorted DataFrame
        """
        # Check if columns exist
        missing_columns = [col for col in self.by if col not in data.columns]
        
        if missing_columns:
            logger.warning(f"Sort columns not found: {', '.join(missing_columns)}")
            # Use only existing columns
            valid_by = [col for col in self.by if col in data.columns]
            valid_ascending = [self.ascending[i] for i, col in enumerate(self.by) if col in data.columns]
            
            if not valid_by:
                logger.error("No valid columns for sorting")
                return data
            
            self.by = valid_by
            self.ascending = valid_ascending
        
        # Sort the DataFrame
        result = data.sort_values(by=self.by, ascending=self.ascending)
        
        # Log sort information
        sort_info = []
        for col, asc in zip(self.by, self.ascending):
            sort_info.append(f"{col} ({'asc' if asc else 'desc'})")
        
        logger.info(f"Sorted data by: {', '.join(sort_info)}")
        
        return result


def get_transform_by_name(transform_type: str, **params) -> Transform:
    """
    Get a transform instance by name.
    
    This helper function creates a transform instance based on the transform type name.
    
    Args:
        transform_type: Name of the transform class
        **params: Parameters to pass to the transform constructor
    
    Returns:
        Transform instance
    
    Raises:
        ValueError: If the transform type is not found
    """
    # Get all transform classes defined in this module
    transforms = {
        'FilterTransform': FilterTransform,
        'TimeWindowTransform': TimeWindowTransform,
        'AggregateTransform': AggregateTransform,
        'PivotTransform': PivotTransform,
        'MetricPivotTransform': MetricPivotTransform,
        'CalculatedMetricTransform': CalculatedMetricTransform,
        'ZScoreOutlierTransform': ZScoreOutlierTransform,
        'NormalizationTransform': NormalizationTransform,
        'MergeTransform': MergeTransform,
        'SortTransform': SortTransform
    }
    
    if transform_type not in transforms:
        raise ValueError(f"Transform type not found: {transform_type}")
    
    # Create an instance of the transform
    transform_class = transforms[transform_type]
    return transform_class(**params)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage
    try:
        # Create sample data
        data = pd.DataFrame({
            'model_name': ['bert', 'bert', 'gpt', 'gpt', 't5', 't5'],
            'hardware_type': ['cpu', 'gpu', 'cpu', 'gpu', 'cpu', 'gpu'],
            'batch_size': [1, 2, 1, 2, 1, 2],
            'metric_name': ['latency', 'throughput', 'latency', 'throughput', 'latency', 'throughput'],
            'metric_value': [100, 200, 150, 300, 120, 250],
            'timestamp': pd.date_range(start='2023-01-01', periods=6)
        })
        
        print("Original data:")
        print(data)
        
        # Apply FilterTransform
        filter_transform = FilterTransform(model_name='bert')
        filtered_data = filter_transform.transform(data)
        
        print("\nFiltered data (bert only):")
        print(filtered_data)
        
        # Apply TimeWindowTransform
        time_transform = TimeWindowTransform(days=7, timestamp_column='timestamp')
        time_filtered_data = time_transform.transform(data)
        
        print("\nTime filtered data (last 7 days):")
        print(time_filtered_data)
        
        # Apply MetricPivotTransform
        pivot_transform = MetricPivotTransform(
            metric_name_column='metric_name',
            metric_value_column='metric_value',
            group_by=['model_name', 'hardware_type', 'batch_size']
        )
        pivoted_data = pivot_transform.transform(data)
        
        print("\nPivoted data (metrics as columns):")
        print(pivoted_data)
        
        # Apply CalculatedMetricTransform
        def calculate_efficiency(row):
            return row['throughput'] / (row['latency'] + 1)
        
        calc_transform = CalculatedMetricTransform(efficiency=calculate_efficiency)
        calculated_data = calc_transform.transform(pivoted_data)
        
        print("\nData with calculated metrics:")
        print(calculated_data)
        
        # Apply NormalizationTransform
        norm_transform = NormalizationTransform(
            columns=['latency', 'throughput'],
            method='min-max'
        )
        normalized_data = norm_transform.transform(pivoted_data)
        
        print("\nNormalized data:")
        print(normalized_data)
        
        # Apply SortTransform
        sort_transform = SortTransform(
            by=['model_name', 'throughput'],
            ascending=[True, False]
        )
        sorted_data = sort_transform.transform(pivoted_data)
        
        print("\nSorted data:")
        print(sorted_data)
        
    except Exception as e:
        logging.error(f"Error in example: {str(e)}")