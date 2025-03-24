
import os
import random
import re

import numpy as np
# For semantic search using sentence transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import polars as pl

from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models import (
    DatetimeTickFormatter, Select, ColumnDataSource, HoverTool, 
    Button, Div, RadioButtonGroup, TextInput, LinearAxis, Range1d,
    FixedTicker
)
from bokeh.plotting import figure
from bokeh.models.tools import BoxZoomTool, PanTool, SaveTool, ResetTool, WheelZoomTool

from mining.estimate import spline_means

# Set a fixed seed for reproducibility
random.seed(42)
np.random.seed(42)

def calculate_post_volume(df, bin_size='1w'):
    """Calculate post volume over time using specified bin size"""
    # Group by time bins and count
    df = df.with_columns(pl.col('createtime').dt.truncate(bin_size).alias('time_bin'))
    volumes = df.group_by('time_bin').count().sort('time_bin')
    
    # Convert to lists for bokeh
    timestamps = volumes['time_bin'].to_list()
    counts = volumes['count'].to_list()
    
    return timestamps, counts

def prepare_source_data(df):
    """Prepare data for a ColumnDataSource with deterministic calculations"""
    if len(df) < 2:
        # Return empty data if not enough points
        return {
            'timestamps': [],
            'stances': [],
            'platform': [],
            'party': [],
            'trend_timestamps': [],
            'trend_means': [],
            'trend_lower': [],
            'trend_upper': [],
            'volume_timestamps': [],
            'volume_counts': []
        }
    
    # Calculate volume data
    volume_timestamps, volume_counts = calculate_post_volume(df)
    
    # Process document text for hover tooltips
    df = df.with_columns(
        pl.when(pl.col('Document').str.len_chars() > 300)\
        .then(pl.col('Document').str.slice(0, 297) + pl.lit('...'))\
        .otherwise(pl.col('Document'))\
        .str.replace("<", "&lt;")\
        .str.replace(">", "&gt;").alias('document_text')
    )
    
    # Calculate trend using polars rolling_mean with dynamic window size
    # Ensure data is sorted by timestamp for rolling calculations
    df_sorted = df.sort('createtime')
    
    # Determine appropriate window size based on data points
    # For smaller datasets use a smaller window, for larger ones use a bigger window
    n_points = len(df_sorted)
    if n_points < 20:
        window_size = max(3, n_points // 2)  # Minimum window size of 3
        window_margin = 1
    elif n_points < 100:
        window_size = n_points // 4
        window_margin = window_size // 4
    else:
        window_size = n_points // 10
        window_margin = window_size // 5
    
    print(f"Using rolling window of size {window_size} with margin {window_margin} for {n_points} points")
    
    # Create a temporary dataframe for rolling calculations
    trend_df = df_sorted.select([
        pl.col('createtime'),
        pl.col('Stance')
    ])
    
    # Calculate rolling mean for trend
    trend_df = trend_df.with_columns([
        pl.col('Stance').rolling_mean(
            window_size=window_size,
            center=True,
            min_periods=1
        ).alias('trend_mean')
    ])
    
    # Calculate standard deviation for confidence intervals
    trend_df = trend_df.with_columns([
        pl.col('Stance').rolling_std(
            window_size=window_size,
            center=True,
            min_periods=1
        ).alias('trend_std')
    ])
    
    # Fill any null values that might exist at the edges
    # trend_df = trend_df.with_columns([
    #     pl.col('trend_mean').forward_fill(),
    #     pl.col('trend_mean').backward_fill(),
    #     pl.col('trend_std').forward_fill(),
    #     pl.col('trend_std').backward_fill()
    # ])
    
    # Calculate confidence intervals (mean ± 1.96 * std for 95% confidence)
    trend_df = trend_df.with_columns([
        (pl.col('trend_mean') - 1.96 * pl.col('trend_std')).clip(-1, 1).alias('trend_lower'),
        (pl.col('trend_mean') + 1.96 * pl.col('trend_std')).clip(-1, 1).alias('trend_upper')
    ])
    
    # Extract arrays for plotting
    trend_timestamps = trend_df['createtime'].to_numpy()
    trend_means = trend_df['trend_mean'].to_numpy()
    trend_lower = trend_df['trend_lower'].to_numpy()
    trend_upper = trend_df['trend_upper'].to_numpy()
        
    
    # Create separate dataframes for the different data components
    scatter_df = df.select(['createtime', 'Stance', 'platform', 'Party', 'document_text', 'SeedName', 'seed_id'])
    scatter_df = scatter_df.rename({'createtime': 'timestamps', 'Stance': 'stances', 'Party': 'party', 'SeedName': 'seed_name'})
    
    trend_df = pl.DataFrame({
        'trend_timestamps': trend_timestamps,
        'trend_means': trend_means,
        'trend_lower': trend_lower,
        'trend_upper': trend_upper
    })
    
    volume_df = pl.DataFrame({
        'volume_timestamps': volume_timestamps,
        'volume_counts': volume_counts
    })
    
    return {
        'scatter_df': scatter_df.to_pandas(),
        'trend_df': trend_df.to_pandas(),
        'volume_df': volume_df.to_pandas()
    }

def create_target_plot(target_df, target_name, unique_platforms, unique_parties):
    """Create a plot for a specific target with filter controls"""
    
    # Create main hover tooltips
    tooltips=[
        ("Name", "@seed_name"),
        ("Content", "@document_text{safe}")
    ]
    
    # Create tools for linking plots
    tools = [
        PanTool(dimensions="width"),
        WheelZoomTool(dimensions="width"),
        BoxZoomTool(dimensions="width"),
        ResetTool(),
        SaveTool()
    ]

    # Create the main stance plot
    p1 = figure(
        title=target_name,
        width=1400,
        height=200,
        x_axis_label="",  # No x-axis label on top plot
        y_axis_label="Stance",
        y_range=(-1, 1),  # Fix y range from -1 to 1
        tools=tools,
        tooltips=tooltips,
    )
    
    # Create the volume subplot
    p2 = figure(
        width=1400,
        height=100,
        x_axis_label="Time",
        y_axis_label="Volume",
        x_range=p1.x_range,  # Link x range
        tools=tools,
    )
    
    # Prepare the source data
    source_data = prepare_source_data(target_df)
    
    # Create separate sources for different components - using pandas
    scatter_source = ColumnDataSource(source_data['scatter_df'])
    trend_source = ColumnDataSource(source_data['trend_df'])
    volume_source = ColumnDataSource(source_data['volume_df'])
    
    # Initial trend line 
    line = p1.line('trend_timestamps', 'trend_means', source=trend_source, line_width=2)
    band = p1.varea('trend_timestamps', 'trend_lower', 'trend_upper', source=trend_source, fill_alpha=0.2)
    
    # Define scatter renderer but with visible=False initially
    scatter = p1.scatter('timestamps', 'stances', source=scatter_source, color="blue", marker='x', size=5, visible=False)
    
    # Volume plot
    volume_line = p2.line('volume_timestamps', 'volume_counts', source=volume_source, line_width=2, color='green')
    volume_area = p2.varea('volume_timestamps', 0, 'volume_counts', source=volume_source, fill_alpha=0.2, fill_color='green')
    
    # Format x-axis
    p1.xaxis[0].formatter = DatetimeTickFormatter(months="%b %Y")
    p2.xaxis[0].formatter = DatetimeTickFormatter(months="%b %Y")
    
    # Set stance axis ticks and labels
    p1.yaxis.ticker = FixedTicker(ticks=[-1, 0, 1])
    p1.yaxis.major_label_overrides = {-1: 'Against', 0: 'Neutral', 1: 'For'}
    
    # Create filter type selector (all, platform, party)
    filter_type_select = Select(
        title="Filter by:",
        options=["all", "platform", "Party"],
        value="all",
        width=150
    )
    
    # Get available platforms and parties for this target
    available_platforms = []
    for platform in unique_platforms:
        platform_df = target_df.filter(pl.col('platform') == platform)
        if len(platform_df) >= 2:
            available_platforms.append(platform)
    
    available_parties = []
    for party in unique_parties:
        party_df = target_df.filter(pl.col('Party') == party)
        if len(party_df) >= 2:
            available_parties.append(party)
    
    # Create attribute selector (initially hidden/disabled)
    attribute_select = Select(
        title="Select value:",
        options=["all"],
        value="all",
        width=150,
        disabled=True
    )
    
    # Create a checkbox for scatter plot visibility
    show_scatter_toggle = RadioButtonGroup(
        labels=["Hide Points", "Show Points"], 
        active=0,
        width=150
    )
    
    # Function to check if time range is less than a month
    def is_time_range_narrow():
        if scatter_source.data['timestamps'] is None or len(scatter_source.data['timestamps']) < 2:
            return False
        
        # Sort timestamps to ensure correct range calculation
        timestamps = pd.Series(scatter_source.data['timestamps']).sort_values()
        if len(timestamps) < 2:
            return False
            
        time_range = timestamps.iloc[-1] - timestamps.iloc[0]
        # Check if range is less than 31 days
        return time_range.total_seconds() < 31 * 24 * 60 * 60
    
    # Server-side callback for filter_type changes
    def filter_type_change(attr, old, new):
        if new == "all":
            attribute_select.options = ["all"]
            attribute_select.value = "all"
            attribute_select.disabled = True
        elif new == "platform":
            attribute_select.options = ["all"] + available_platforms
            attribute_select.value = "all"
            attribute_select.disabled = False
        elif new == "Party":
            attribute_select.options = ["all"] + available_parties
            attribute_select.value = "all"
            attribute_select.disabled = False
            
        update_plot(filter_type_select.value, attribute_select.value)
    
    # Server-side callback for attribute changes
    def attribute_change(attr, old, new):
        update_plot(filter_type_select.value, new)
    
    # Callback for scatter visibility toggle
    def toggle_scatter(attr, old, new):
        scatter.visible = (new == 1)
        
    show_scatter_toggle.on_change('active', toggle_scatter)
    
    # Function to update the plot based on filter selections
    def update_plot(filter_type, attribute_value):
        filtered_df = target_df
        
        if filter_type != "all" and attribute_value != "all":
            if filter_type == "platform":
                filtered_df = target_df.filter(pl.col('platform') == attribute_value)
            elif filter_type == "Party":
                filtered_df = target_df.filter(pl.col('Party') == attribute_value)
        
        # Only update if we have enough data points
        if len(filtered_df) >= 2:
            new_data = prepare_source_data(filtered_df)
            
            # Update sources with pandas DataFrames
            scatter_source.data = ColumnDataSource.from_df(new_data['scatter_df'])
            trend_source.data = ColumnDataSource.from_df(new_data['trend_df'])
            volume_source.data = ColumnDataSource.from_df(new_data['volume_df'])
            
            # Automatically show scatter only for narrow time ranges
            time_range_narrow = is_time_range_narrow()
            if time_range_narrow:
                scatter.visible = (show_scatter_toggle.active == 1)
            else:
                scatter.visible = False  # Always hide for wide time ranges
                
            # Update the toggle label based on the time range
            if not time_range_narrow and show_scatter_toggle.active == 1:
                show_scatter_toggle.active = 0  # Reset to "Hide Points"
            
            print(f"Updated plot for {target_name}: {filter_type}={attribute_value}, {len(filtered_df)} data points")
    
    # Connect callbacks
    filter_type_select.on_change('value', filter_type_change)
    attribute_select.on_change('value', attribute_change)
    
    # Create a combined plot
    plots = gridplot([[p1], [p2]], toolbar_location="right", sizing_mode="fixed")
    
    # Create a row for this target with the plot and selectors
    target_row = row(
        plots,
        column(filter_type_select, attribute_select, show_scatter_toggle),
    )
    
    return target_row

class StanceDashboard:
    def __init__(self, doc):
        self.doc = doc
        self.targets_per_page = 5
        self.current_page = 0
        self.all_targets = []
        self.filtered_targets = []  # For search results
        self.total_pages = 0
        self.dashboard_layout = None
        self.pagination_controls = None
        self.page_info = None
        self.search_mode = False  # Track if we're in search mode
        
        # Load data
        self.load_data()
        
        # Load the sentence transformer model
        self.load_model()
        
        # Initialize dashboard
        self.initialize_dashboard()
    
    def load_model(self):
        """Load the sentence transformer model for semantic search"""
        try:
            print("Loading sentence transformer model...")
            # Skip model loading since it's too slow (as per your request)
            raise Exception("Too slow")
            # Use a smaller model for faster loading
            self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            
            # Pre-compute embeddings for all targets for faster search
            target_names = [target['Target'] for target in self.all_targets]
            self.target_embeddings = self.model.encode(target_names, show_progress_bar=True)
            print(f"Computed embeddings for {len(target_names)} targets")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.target_embeddings = None
    
    def load_data(self):
        # Load data with fewer columns
        dir_path = './data/stance_targets/'
        df = pl.DataFrame()
        for filename in os.listdir(dir_path):
            if re.search(r'\d{4}_\d{1,2}_doc_targets_with_stance.parquet.zstd', filename):
                file_df = pl.read_parquet(
                    os.path.join(dir_path, filename),
                    # Specify only the columns we need to improve loading time
                    columns=['id', 'createtime', 'Document', 'Targets', 'Polarities', 'seed_id']
                )
                df = pl.concat([df, file_df], how='diagonal_relaxed')

        # Use a faster method to determine platform
        df = df.with_columns(
            pl.when(pl.col('id').is_null())\
            .then(pl.lit('tiktok'))\
            .when(pl.col('id').str.contains('_'))\
            .then(pl.lit('instagram'))\
            .otherwise(pl.lit('twitter'))\
            .alias('platform')
        )
        
        # Cast once, not in a loop
        df = df.with_columns(pl.col('seed_id').cast(pl.Int64))
        
        # Join with seedlist - only get needed columns
        seedlist_df = pl.read_parquet(
            './data/seedlist.parquet.zstd',
            columns=['SeedID', 'SeedName', 'Party']
        )
        df = df.join(seedlist_df, left_on='seed_id', right_on='SeedID')
        df = df.with_columns(pl.col('Party').fill_null('None'))  # faster than replace
        
        # Explode targets and polarities with error handling
        try:
            df = df.explode(['Targets', 'Polarities']).rename({'Targets': 'Target', 'Polarities': 'Stance'})
        except pl.exceptions.ShapeError:
            # Filter first, then explode
            df = df.filter(pl.col('Targets').list.len() == pl.col('Polarities').list.len())
            df = df.explode(['Targets', 'Polarities']).rename({'Targets': 'Target', 'Polarities': 'Stance'})

        # Entity mapping for standardizing names
        entity_mapping = {
            'Justin Trudeau': ['trudeau', 'justin trudeau', '@justintrudeau'],
            'Mark Carney': ['carney', 'mark carney', 'mark j. carney', 'mark j carney']
        }
        entity_replace = {k: v for v, k_list in entity_mapping.items() for k in k_list}
        df = df.with_columns(pl.col('Target').replace(entity_replace))
        
        # Get ordered list of all targets
        target_count_df = df.group_by('Target').len().sort('len', descending=True)
        self.all_targets = target_count_df.to_dicts()
        self.filtered_targets = self.all_targets.copy()  # Initialize filtered targets
        self.total_pages = (len(self.all_targets) + self.targets_per_page - 1) // self.targets_per_page
        
        # Store the dataframe for later use
        self.df = df
        
        # Get unique values for our filters
        self.unique_platforms = sorted(df['platform'].unique().to_list())
        self.unique_parties = sorted(df['Party'].unique().to_list())
    
    def semantic_search(self, query, top_k=None):
        """Perform semantic search on target names using sentence transformers"""
        if not self.model or not self.target_embeddings:
            print("Model not loaded, using fallback text search")
            return self.text_search(query)
            
        try:
            # Encode the query
            query_embedding = self.model.encode([query])[0]
            
            # Calculate similarity scores
            similarities = cosine_similarity(
                [query_embedding], 
                self.target_embeddings
            )[0]
            
            # Create (target, similarity) pairs
            target_scores = list(zip(self.all_targets, similarities))
            
            # Sort by similarity (descending)
            target_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top_k results or all if top_k is None
            if top_k:
                target_scores = target_scores[:top_k]
            
            # Filter for targets with meaningful similarity
            target_scores = [(target, score) for target, score in target_scores if score > 0.2]
            
            # Extract just the targets
            results = [target for target, score in target_scores]
            
            print(f"Semantic search results for '{query}': {len(results)} matches")
            return results
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return self.text_search(query)  # Fallback
    
    def text_search(self, query):
        """Fallback text search if semantic search fails"""
        query = query.lower()
        results = [
            target for target in self.all_targets 
            if query in target['Target'].lower()
        ]
        return results
    
    def initialize_dashboard(self):
        # Create search controls
        self.create_search_controls()
        
        # Create pagination controls
        self.create_pagination_controls()
        
        # Generate initial page
        self.update_page(0)
    
    def create_search_controls(self):
        # Create search input
        self.search_input = TextInput(
            title="Search stance targets:",
            placeholder="Enter search terms...",
            width=400
        )
        
        # Create search button
        self.search_button = Button(
            label="Search",
            button_type="primary",
            width=100
        )
        
        # Create clear button
        self.clear_button = Button(
            label="Clear",
            button_type="default",
            width=100
        )
        
        # Set up callbacks
        self.search_button.on_click(self.perform_search)
        self.clear_button.on_click(self.clear_search)
        self.search_input.on_change('value', lambda attr, old, new: self.search_input_change())
        
        # Create search controls row
        self.search_controls = row(
            self.search_input,
            self.search_button,
            self.clear_button,
            width=600,
            styles={'margin-bottom': '20px'}
        )
        
        # Create search results info div
        self.search_results_info = Div(
            text="",
            width=600,
            styles={'margin-bottom': '10px', 'color': 'blue'}
        )
    
    def search_input_change(self):
        """Enable the search button when text is entered"""
        if self.search_input.value.strip():
            self.search_button.disabled = False
        else:
            self.search_button.disabled = True
    
    def perform_search(self):
        """Execute the search and update the dashboard"""
        query = self.search_input.value.strip()
        if not query:
            return
        
        # Perform semantic search
        search_results = self.semantic_search(query)
        
        if search_results:
            # Update filtered targets
            self.filtered_targets = search_results
            self.search_mode = True
            
            # Update pagination for search results
            self.total_pages = (len(self.filtered_targets) + self.targets_per_page - 1) // self.targets_per_page
            
            # Update search results info
            self.search_results_info.text = f"Found {len(search_results)} targets matching '{query}'"
            
            # Go to first page of results
            self.update_page(0)
        else:
            # No results found
            self.search_results_info.text = f"No targets found matching '{query}'"
    
    def clear_search(self):
        """Clear search and show all targets"""
        self.search_input.value = ""
        self.search_results_info.text = ""
        self.filtered_targets = self.all_targets.copy()
        self.search_mode = False
        
        # Reset pagination
        self.total_pages = (len(self.all_targets) + self.targets_per_page - 1) // self.targets_per_page
        
        # Go to first page
        self.update_page(0)
    
    def create_pagination_controls(self):
        # Create pagination controls
        prev_button = Button(label="Previous", width=100, button_type="default", disabled=True)
        next_button = Button(label="Next", width=100, button_type="default")
        
        # Page information display
        self.page_info = Div(
            text=f"Page {self.current_page + 1} of {self.total_pages}",
            width=150,
            styles={'font-size': '16px', 'text-align': 'center'}
        )
        
        # Connect callbacks
        prev_button.on_click(self.go_to_prev_page)
        next_button.on_click(self.go_to_next_page)
        
        # Store buttons for enabling/disabling
        self.prev_button = prev_button
        self.next_button = next_button
        
        # Create pagination row
        pagination_row = row(
            prev_button, 
            self.page_info, 
            next_button,
            sizing_mode="fixed",
            styles={'margin-top': '20px', 'margin-bottom': '20px', 'text-align': 'center'}
        )
        
        # Store pagination controls
        self.pagination_controls = pagination_row
    
    def go_to_prev_page(self):
        if self.current_page > 0:
            self.update_page(self.current_page - 1)
    
    def go_to_next_page(self):
        if self.current_page < self.total_pages - 1:
            self.update_page(self.current_page + 1)
    
    def update_page(self, page_number):
        # Store current page
        self.current_page = page_number
        
        # Calculate start and end indices
        start_idx = page_number * self.targets_per_page
        end_idx = min(start_idx + self.targets_per_page, len(self.filtered_targets))
        
        # Get targets for this page (from filtered targets)
        page_targets = self.filtered_targets[start_idx:end_idx]
        
        # Create plots for each target
        plot_rows = []
        for target in page_targets:
            target_name = target['Target']
            target_df = self.df.filter(self.df['Target'] == target_name)
            
            # Create a plot for this target
            target_row = create_target_plot(target_df, target_name, self.unique_platforms, self.unique_parties)
            plot_rows.append(target_row)
        
        # Update page information
        self.page_info.text = f"Page {self.current_page + 1} of {self.total_pages}"
        
        # Enable/disable navigation buttons
        self.prev_button.disabled = (self.current_page == 0)
        self.next_button.disabled = (self.current_page == self.total_pages - 1)
        
        # Create dashboard layout
        title = Div(
            text=f"<h1>Stance Target Dashboard</h1>",
            width=1600,
            styles={'text-align': 'center', 'margin-bottom': '10px'}
        )
        
        # Create the final layout
        dashboard_content = column(
            title,
            row(self.search_controls, width=1600, styles={'justify-content': 'center'}),
            row(self.search_results_info, width=1600, styles={'justify-content': 'center'}),
            row(self.pagination_controls, width=1600, styles={'justify-content': 'center', 'align-items': 'center'}),
            column(plot_rows) if plot_rows else Div(text="<h3>No targets to display</h3>", width=1600, styles={'text-align': 'center', 'margin-top': '50px'}),
            row(self.pagination_controls, width=1600, styles={'justify-content': 'center', 'align-items': 'center'}),
            width=1600
        )
        
        # Remove any existing layout
        if self.dashboard_layout:
            self.doc.remove_root(self.dashboard_layout)
        
        # Add the new layout
        self.dashboard_layout = dashboard_content
        self.doc.add_root(self.dashboard_layout)
        self.doc.title = "Stance Target Dashboard"

def main(doc):
    # Create dashboard
    dashboard = StanceDashboard(doc)

# This is used when running the script with bokeh serve
curdoc().add_next_tick_callback(lambda: main(curdoc()))