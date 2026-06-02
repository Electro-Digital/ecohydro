# ============================================================================
# CBK Weather Forecasting Dashboard - Flask Application
# ============================================================================
# This Flask application provides a weather monitoring and forecasting system
# for power plants (Botocan and Kalayaan). It includes user authentication,
# role-based access control, and real-time data visualization.
# ============================================================================

# Import required Flask modules and extensions
from flask import Flask, render_template, request, redirect, flash, url_for, jsonify, session
from flask_wtf import FlaskForm  # For secure form handling
from wtforms import StringField, PasswordField, SubmitField  # Form field types
from wtforms.validators import DataRequired  # Form validation
from flask_mysqldb import MySQL  # MySQL database integration
from functools import wraps  # For creating decorators
import random
import time
import datetime
import math
import smtplib
import secrets
import hashlib
#from email.mime.text import MIMEText
#from email.mime.multipart import MIMEMultipart
from db_config import MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB  # Database credentials
#from email_config import SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, SMTP_SENDER_NAME, OTP_EMAIL_TEMPLATE, WELCOME_EMAIL_TEMPLATE
from forecast_engine import ForecastEngine
import pandas as pd

# ============================================================================
# APPLICATION SETUP
# ============================================================================

# Initialize Flask application
app = Flask(__name__)
# Secret key for session management and CSRF protection
app.secret_key = 'your-secret-key-change-this-in-production'

# Configure MySQL database connection
app.config['MYSQL_HOST'] = MYSQL_HOST
app.config['MYSQL_USER'] = MYSQL_USER
app.config['MYSQL_PASSWORD'] = MYSQL_PASSWORD
app.config['MYSQL_DB'] = MYSQL_DB
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'  # Return results as dictionaries

# Initialize MySQL connection
mysql = MySQL(app)

# Initialize Forecast Engine
forecast_engine = ForecastEngine()

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Maps location IDs to their corresponding database table names
# This allows us to dynamically query the correct table based on user selection
LOCATION_TABLE_MAP = {
    'botocan1': 'b01_parameters',      # Botocan plant data table
    'botocan2': 'b02_parameters',     # Botocan plant 2 data table
    # KALAYAAN LOCATIONS - COMMENTED OUT FOR BOTOCAN-ONLY FOCUS
    # 'kalayaan01': 'k01_parameters',   # Kalayaan unit 1 data table
    # 'kalayaan02': 'k02_parameters',   # Kalayaan unit 2 data table
    # 'kalayaan03': 'k03_parameters',   # Kalayaan unit 3 data table
    # 'kalayaan04': 'k04_parameters'    # Kalayaan unit 4 data table
}

# List of all available locations for dropdown menus and access control
# Each location has an ID (used in code) and a display name (shown to users)
AVAILABLE_LOCATIONS = [
    # {'id': 'none', 'name': 'None'},                    # No location selected - COMMENTED OUT
    {'id': 'botocan1', 'name': 'Botocan Unit 1'},       # Botocan hydroelectric plant unit 1 - DEFAULT
    {'id': 'botocan2', 'name': 'Botocan Unit 2'},      # Botocan hydroelectric plant unit 2
    # KALAYAAN LOCATIONS - COMMENTED OUT FOR BOTOCAN-ONLY FOCUS
    # {'id': 'kalayaan01', 'name': 'Kalayaan 01'},       # Kalayaan pumped storage unit 1
    # {'id': 'kalayaan02', 'name': 'Kalayaan 02'},       # Kalayaan pumped storage unit 2
    # {'id': 'kalayaan03', 'name': 'Kalayaan 03'},       # Kalayaan pumped storage unit 3
    # {'id': 'kalayaan04', 'name': 'Kalayaan 04'},       # Kalayaan pumped storage unit 4
]

# ============================================================================
# FORM CLASSES
# ============================================================================

# Login form class using Flask-WTF for secure form handling
# This provides CSRF protection and server-side validation
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])  # Required username field
    password = PasswordField('Password', validators=[DataRequired()])  # Required password field
    submit = SubmitField('Login')  # Submit button

# ============================================================================
# SECURITY DECORATORS
# ============================================================================

# Decorator to require user login for protected routes
# This checks if user is logged in by verifying session data
def login_required(f):
    @wraps(f)  # Preserves original function metadata
    def decorated_function(*args, **kwargs):
        # Check if user_id exists in session (user is logged in)
        if 'user_id' not in session:
            return redirect(url_for('login'))  # Redirect to login if not authenticated
        return f(*args, **kwargs)  # Call original function if authenticated
    return decorated_function

# Decorator to require admin privileges for certain routes
# This provides an additional layer of access control
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if user is logged in AND has admin role
        if 'user_id' not in session or session.get('role') != 'admin':
            flash('Admin access required.', 'danger')  # Show error message
            return redirect(url_for('dashboard'))  # Redirect to dashboard
        return f(*args, **kwargs)
    return decorated_function

# ============================================================================
# ACCESS CONTROL FUNCTIONS
# ============================================================================

# Get the list of locations the current user has access to
# Returns empty list if no locations are assigned
def get_user_locations():
    return session.get('locations', [])

# Check if the current user can access a specific location
# This implements the business logic for location-based access control
def can_access_location(location):
    user_locations = get_user_locations()
    
    # Admin users with 'all' access can view everything
    if 'all' in user_locations:
        return True
    
    # Direct access: user has explicit permission for this location
    if location in user_locations:
        return True
    
    # Group-based access: define location groups
    # KALAYAAN LOCATIONS - COMMENTED OUT FOR BOTOCAN-ONLY FOCUS
    # kalayaan_locations = ['kalayaan01', 'kalayaan02', 'kalayaan03', 'kalayaan04']
    botocan_locations = ['botocan1', 'botocan2']
    
    # KALAYAAN ACCESS CHECK - COMMENTED OUT FOR BOTOCAN-ONLY FOCUS
    # if location in kalayaan_locations:
    #     return any(loc in user_locations for loc in kalayaan_locations)
    
    # If requesting Botocan location or aggregate, check if user has ANY Botocan access
    if location in botocan_locations or location == 'botocan_all':
        return any(loc in user_locations for loc in botocan_locations)
    
    # Default: deny access
    return False

# ============================================================================
# ROUTE HANDLERS
# ============================================================================

# Root route - redirects to login page
# This ensures users always start at the login screen
@app.route('/')
def index():
    return redirect(url_for('login'))

# Login route - handles both GET (show form) and POST (process login)
@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()  # Create login form instance

    # Process form submission (POST request)
    if form.validate_on_submit():
        username = form.username.data  # Get username from form
        password = form.password.data  # Get password from form

        try:
            # Query database for user credentials
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM system_users WHERE username = %s", (username,))
            user = cur.fetchone()  # Get user record
            cur.close()

            # Verify user exists and password matches
            if user and user['password'] == password:
                # Store user information in session for future requests
                session['user_id'] = user['username']
                session['role'] = user['role']  # Store user role (admin, user, etc.)
                # Parse comma-separated locations string into list
                session['locations'] = user['locations'].split(',') if user['locations'] else []
                flash('Login successful!', 'success')
                # Redirect to main dashboard after successful login
                return redirect(url_for('main_dashboard'))
            else:
                flash('Invalid credentials.', 'danger')
        except Exception as e:
            # Handle database connection errors gracefully
            flash('Database connection error.', 'danger')
            print(f"Login error: {e}")

    # Show login form (GET request or failed POST)
    return render_template('login.html', form=form)

@app.route('/dashboard')
@login_required
def dashboard():
    user_locations = get_user_locations()
    is_admin = session.get('role') == 'admin'
    
    if is_admin:
        available_locations = AVAILABLE_LOCATIONS
    else:
        # KALAYAAN LOCATIONS - COMMENTED OUT FOR BOTOCAN-ONLY FOCUS
        # kalayaan_locations = ['kalayaan01', 'kalayaan02', 'kalayaan03', 'kalayaan04']
        botocan_locations = ['botocan1', 'botocan2']
        
        # KALAYAAN ACCESS CHECK - COMMENTED OUT FOR BOTOCAN-ONLY FOCUS
        # has_kalayaan_access = any(loc in user_locations for loc in kalayaan_locations)
        has_botocan_access = any(loc in user_locations for loc in botocan_locations)
        
        available_locations = []
        
        for loc in AVAILABLE_LOCATIONS:
            if loc['id'] in user_locations:
                available_locations.append(loc)
            # KALAYAAN ACCESS CHECK - COMMENTED OUT FOR BOTOCAN-ONLY FOCUS
            # elif has_kalayaan_access and loc['id'] in kalayaan_locations:
            #     available_locations.append(loc)
            elif has_botocan_access and loc['id'] in botocan_locations:
                available_locations.append(loc)
    
    return render_template('admin_dashboard.html', 
                         is_admin=is_admin,
                         available_locations=available_locations,
                         user_locations=user_locations)

@app.route('/admin_dashboard')
@login_required
def admin_dashboard():
    return redirect(url_for('dashboard'))

# ============================================================================
# DATA RETRIEVAL FUNCTIONS
# ============================================================================

# Fetch the most recent real data from database for a specific location
# This is the core function that retrieves live plant data
def get_real_data_from_db(location):
    try:
        # Get the correct table name for this location
        table_name = LOCATION_TABLE_MAP.get(location)
        if not table_name:
            return None  # Location not found in mapping

        # Execute database query to get latest record
        cur = mysql.connection.cursor()
        query = f"SELECT * FROM {table_name} ORDER BY ts DESC LIMIT 1"  # Get most recent record
        cur.execute(query)
        result = cur.fetchone()
        cur.close()

        if result:
            # Transform database result into standardized format
            # This ensures consistent data structure across all locations
            data = {
                # Timestamp and location info
                'ts': result.get('ts', time.strftime('%Y-%m-%d %H:%M:%S')),
                'location': location,
                
                # Power generation data
                'mw': result.get('mw', 0),  # Total megawatts
                'mw_turbine': result.get('mw_turbine', result.get('mw', 0)),  # Turbine power
                'mw_pump': result.get('mw_pump', 0),  # Pump power (for pumped storage)
                'mvar': result.get('mvar', 0),  # Reactive power
                
                # Electrical parameters
                'voltage': result.get('voltage', 0),
                'current': result.get('current', 0),
                'hz': result.get('hz', 50.0),  # Frequency (default 50Hz)
                
                # Water flow data
                'flow_turbine': result.get('flow_turbine', 0),
                'flow_pump': result.get('flow_pump', 0),
                'flow_other': result.get('flow_other', 0),
                'net_head': result.get('net_head', 0),  # Water head pressure
                
                # Water levels and temperature
                'temp_water': result.get('temp_water', 0),
                'upper_water1': result.get('upper_water1', result.get('upper_water', 0)),
                'upper_water2': result.get('upper_water2', 0),
                'lower_water1': result.get('lower_water1', 0),
                'lower_water2': result.get('lower_water2', 0),
                'evaporation': result.get('evaporation', 0),
                
                # Weather data (using standard meteorological codes)
                'air_temp': result.get('t', 0),        # Temperature
                'humidity': result.get('rh', 0),       # Relative humidity
                'rainfall': result.get('rr', 0),       # Rainfall
                'pressure': result.get('sp', 0),       # Surface pressure
                'windspeed': result.get('ws', 0),      # Wind speed
                'wind_direction': result.get('wd', 0), # Wind direction
                'solar_radiation': result.get('sr', 0), # Solar radiation
                
                'data_source': 'database'  # Mark data source for debugging
            }
            return data
        return None  # No data found
    except Exception as e:
        print(f"Database error for location {location}: {e}")
        return None

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully.', 'success')
    return redirect(url_for('login'))



@app.route('/api/sensor-data')
@login_required
def get_sensor_data():
    # CHANGED DEFAULT FROM KALAYAAN01 TO BOTOCAN FOR BOTOCAN-ONLY FOCUS
    location = request.args.get('location', 'botocan')
    
    if not can_access_location(location):
        return jsonify({'error': 'Access denied to this location'}), 403

    # COMMENTED OUT 'ALL' OPTION FOR BOTOCAN-ONLY FOCUS
    # if location == 'all':
    #     return jsonify(get_aggregated_data())
    if location == 'botocan_all':
        return jsonify(get_botocan_aggregated_data())
    elif location == 'none':
        return jsonify(get_empty_data())

    real_data = get_real_data_from_db(location)
    if real_data:
        return jsonify(real_data)
    else:
        # Return simulated data when real data is not available
        return jsonify(get_location_specific_data(location))

@app.route('/api/user-info')
@login_required
def get_user_info():
    return jsonify({
        'user_id': session.get('user_id'),
        'role': session.get('role'),
        'locations': session.get('locations'),
        'is_admin': session.get('role') == 'admin'
    })

# ============================================================================
# API ENDPOINTS
# ============================================================================

# API endpoint for weather forecast data
# This provides 7-day weather forecast data for the dashboard charts
@app.route('/api/weather-forecast')
@login_required  # Require user authentication
def get_weather_forecast():
    # Extract query parameters with defaults
    location = request.args.get('location', 'botocan')  # Default to Botocan
    grouping = request.args.get('grouping', 'hourly')   # Default to hourly data
    # Parse comma-separated parameters list
    parameters = request.args.get('parameters', 't').split(',') if request.args.get('parameters') else ['t']
    start_date = request.args.get('start_date')  # Optional custom date range
    end_date = request.args.get('end_date')
    
    try:
        # Map location to appropriate weather forecast table
        if location == 'botocan':
            table_name = 'openmeteo_bhepp_7d'
        # KALAYAAN WEATHER TABLE - COMMENTED OUT FOR BOTOCAN-ONLY FOCUS
        # elif location == 'kalayaan':
        #     table_name = 'openmeteo_kpspp_7d'
        else:
            return jsonify({'error': 'Invalid location'}), 400
        
        cur = mysql.connection.cursor()
        
        # Build date filter
        if start_date and end_date:
            date_filter = f"ts BETWEEN '{start_date}' AND '{end_date}'"
        else:
            date_filter = "ts >= DATE_SUB(NOW(), INTERVAL 7 DAY)"
        
        query = f"""
            SELECT ts as timestamp, t, rh, ws, sr, rr, sp, wd
            FROM {table_name} 
            WHERE {date_filter}
            ORDER BY ts ASC
        """
        
        cur.execute(query)
        results = cur.fetchall()
        cur.close()
        
        if not results:
            return jsonify({'error': 'No forecast data available'}), 404
        
        # Process data
        labels = []
        data = {param: [] for param in parameters}
        
        for row in results:
            timestamp = row['timestamp']
            labels.append(timestamp.strftime('%b %d %H:%M'))
            
            for param in parameters:
                value = row.get(param)
                if value is None:
                    data[param].append(0)
                else:
                    data[param].append(float(value))
        
        response = {'labels': labels}
        response.update(data)
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Weather forecast error: {e}")
        return jsonify({'error': 'Database error'}), 500

def get_aggregated_data():
    all_locations = ['botocan1', 'botocan2', 'kalayaan01', 'kalayaan02', 'kalayaan03', 'kalayaan04']
    total_mw = 0
    total_mvar = 0
    count = 0
    
    for loc in all_locations:
        data = get_real_data_from_db(loc)
        if data:
            total_mw += data.get('mw', 0)
            total_mvar += data.get('mvar', 0)
            count += 1

    return {
        'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
        'location': 'all',
        'mw': total_mw,
        'mvar': total_mvar,
        'data_source': 'aggregated'
    }

def get_botocan_aggregated_data():
    botocan_locations = ['botocan1', 'botocan2']
    total_mw = 0
    total_mvar = 0
    count = 0
    
    for loc in botocan_locations:
        data = get_real_data_from_db(loc)
        if data:
            total_mw += data.get('mw', 0)
            total_mvar += data.get('mvar', 0)
            count += 1

    return {
        'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
        'location': 'botocan_all',
        'mw': total_mw,
        'mvar': total_mvar,
        'data_source': 'botocan_aggregated'
    }

def get_empty_data():
    return {
        'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
        'location': 'none',
        'mw': 0, 'mvar': 0,
        'data_source': 'none'
    }

def get_location_specific_data(location):
    """Generate realistic simulated data for specific locations"""
    import random
    
    # Location-specific base configurations
    location_configs = {
        'botocan1': {
            'base_mw': 52.8,
            'base_voltage': 14200,
            'base_flow': 140.2,
            'base_temp': 24.5
        },
        'botocan2': {
            'base_mw': 48.3,
            'base_voltage': 13800,
            'base_flow': 128.7,
            'base_temp': 23.8
        }
    }
    
    config = location_configs.get(location, location_configs['botocan'])
    
    # Generate realistic variations
    return {
        'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
        'location': location,
        
        # Power generation data with realistic variations
        'mw': round(config['base_mw'] + random.uniform(-8, 8), 1),
        'mw_turbine': round(config['base_mw'] * 1.1 + random.uniform(-5, 5), 1),
        'mw_pump': round(random.uniform(2, 6), 1),
        'mvar': round(random.uniform(8, 18), 1),
        
        # Electrical parameters
        'voltage': round(config['base_voltage'] + random.uniform(-300, 300)),
        'current': round(random.uniform(1100, 1400)),
        'hz': round(50.0 + random.uniform(-0.3, 0.3), 1),
        
        # Water flow data
        'flow_turbine': round(config['base_flow'] + random.uniform(-15, 15), 1),
        'flow_pump': round(random.uniform(12, 20), 1),
        'flow_other': round(random.uniform(6, 12), 1),
        'net_head': round(random.uniform(80, 95), 1),
        
        # Water levels and temperature
        'temp_water': round(config['base_temp'] + random.uniform(-2, 2), 1),
        'upper_water1': round(random.uniform(60, 70), 1),
        'upper_water2': round(random.uniform(59, 69), 1),
        'lower_water1': round(random.uniform(42, 48), 1),
        'lower_water2': round(random.uniform(41, 47), 1),
        'evaporation': round(random.uniform(1.5, 3.5), 1),
        
        # Weather data
        'air_temp': round(random.uniform(25, 32), 1),
        'humidity': round(random.uniform(55, 75)),
        'rainfall': round(max(0, random.uniform(-0.5, 3)), 1),
        'pressure': round(random.uniform(1008, 1018), 1),
        'windspeed': round(random.uniform(8, 18), 1),
        'wind_direction': round(random.uniform(0, 360)),
        'solar_radiation': round(random.uniform(700, 950)),
        
        'data_source': 'simulated'
    }

@app.route('/dashboard2')
@login_required
def dashboard2():
    user_locations = get_user_locations()
    is_admin = session.get('role') == 'admin'
    
    if is_admin:
        available_locations = AVAILABLE_LOCATIONS
    else:
        # KALAYAAN LOCATIONS - COMMENTED OUT FOR BOTOCAN-ONLY FOCUS
        # kalayaan_locations = ['kalayaan01', 'kalayaan02', 'kalayaan03', 'kalayaan04']
        botocan_locations = ['botocan1', 'botocan2']
        
        # KALAYAAN ACCESS CHECK - COMMENTED OUT FOR BOTOCAN-ONLY FOCUS
        # has_kalayaan_access = any(loc in user_locations for loc in kalayaan_locations)
        has_botocan_access = any(loc in user_locations for loc in botocan_locations)
        
        available_locations = []
        
        for loc in AVAILABLE_LOCATIONS:
            if loc['id'] in user_locations:
                available_locations.append(loc)
            # KALAYAAN ACCESS CHECK - COMMENTED OUT FOR BOTOCAN-ONLY FOCUS
            # elif has_kalayaan_access and loc['id'] in kalayaan_locations:
            #     available_locations.append(loc)
            elif has_botocan_access and loc['id'] in botocan_locations:
                available_locations.append(loc)
    
    return render_template('admin_dashboard2.html', 
                         is_admin=is_admin,
                         available_locations=available_locations,
                         user_locations=user_locations)

@app.route('/api/chart-data')
@login_required
def get_chart_data():
    # CHANGED DEFAULT FROM KALAYAAN01 TO BOTOCAN FOR BOTOCAN-ONLY FOCUS
    location = request.args.get('location', 'botocan')
    parameter = request.args.get('parameter', 'mw')
    range_type = request.args.get('range', '24h')
    grouping = request.args.get('grouping', 'daily')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    if not can_access_location(location):
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        table_map = {
            'botocan1': 'b01_parameters_hourly',
            'botocan2': 'b02_parameters_hourly',
            # KALAYAAN TABLES - COMMENTED OUT FOR BOTOCAN-ONLY FOCUS
            # 'kalayaan01': 'k01_parameters_hourly', 
            # 'kalayaan02': 'k02_parameters_hourly',
            # 'kalayaan03': 'k03_parameters_hourly',
            # 'kalayaan04': 'k04_parameters_hourly'
        }
        
        table_name = table_map.get(location)
        if not table_name:
            return jsonify({'error': 'Invalid location'}), 400
            
        cur = mysql.connection.cursor()
        
        # Build date filter
        if range_type == 'custom' and start_date and end_date:
            date_filter = f"ts BETWEEN '{start_date}' AND '{end_date} 23:59:59'"
        else:
            if range_type == '24h':
                date_filter = "ts >= DATE_SUB(NOW(), INTERVAL 24 HOUR)"
            elif range_type == '7d':
                date_filter = "ts >= DATE_SUB(NOW(), INTERVAL 7 DAY)"
            elif range_type == '30d':
                date_filter = "ts >= DATE_SUB(NOW(), INTERVAL 30 DAY)"
            else:
                date_filter = "ts >= DATE_SUB(NOW(), INTERVAL 24 HOUR)"
        
        # Build grouping
        if grouping == 'hourly':
            group_by = "DATE_FORMAT(ts, '%Y-%m-%d %H:00:00')"
            time_format = "%Y-%m-%d %H:00:00"
            select_time = "DATE_FORMAT(MIN(ts), '%Y-%m-%d %H:00:00')"
        elif grouping == 'daily':
            group_by = "DATE(ts)"
            time_format = "%Y-%m-%d"
            select_time = "DATE_FORMAT(MIN(ts), '%Y-%m-%d')"
        elif grouping == 'weekly':
            group_by = "YEARWEEK(ts, 1)"
            time_format = "%Y-%u"
            select_time = "YEARWEEK(MIN(ts), 1)"
        elif grouping == 'monthly':
            group_by = "YEAR(ts), MONTH(ts)"
            time_format = "%Y-%m"
            select_time = "DATE_FORMAT(MIN(ts), '%Y-%m')"
        elif grouping == 'yearly':
            group_by = "YEAR(ts)"
            time_format = "%Y"
            select_time = "YEAR(MIN(ts))"
        else:
            group_by = "DATE(ts)"
            time_format = "%Y-%m-%d"
            select_time = "DATE_FORMAT(MIN(ts), '%Y-%m-%d')"
            
        if grouping == 'monthly':
            query = f"""
                SELECT YEAR(ts) as year_val, MONTH(ts) as month_val,
                       AVG({parameter}) as avg_value,
                       {select_time} as formatted_time
                FROM {table_name} 
                WHERE {date_filter} AND {parameter} IS NOT NULL
                GROUP BY {group_by}
                ORDER BY year_val, month_val
            """
        elif grouping == 'yearly':
            query = f"""
                SELECT YEAR(ts) as year_val,
                       AVG({parameter}) as avg_value,
                       {select_time} as formatted_time
                FROM {table_name} 
                WHERE {date_filter} AND {parameter} IS NOT NULL
                GROUP BY {group_by}
                ORDER BY year_val
            """
        else:
            query = f"""
                SELECT {group_by} as time_group, 
                       AVG({parameter}) as avg_value,
                       {select_time} as formatted_time
                FROM {table_name} 
                WHERE {date_filter} AND {parameter} IS NOT NULL
                GROUP BY {group_by}
                ORDER BY time_group
            """
        
        cur.execute(query)
        results = cur.fetchall()
        cur.close()
        
        labels = []
        data = []
        
        if range_type == 'custom' and start_date and end_date:
            from datetime import datetime, timedelta
            import calendar
            
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            if grouping == 'hourly':
                # Generate all hours in the range
                data_dict = {}
                for row in results:
                    key = row['formatted_time']
                    data_dict[key] = row['avg_value'] if row['avg_value'] is not None else 0
                
                current = start
                while current <= end:
                    key = current.strftime('%Y-%m-%d %H:00:00')
                    labels.append(current.strftime('%m/%d %H:00'))
                    data.append(data_dict.get(key, 0))
                    current += timedelta(hours=1)
                    
            elif grouping == 'daily':
                # Generate all days in the range
                data_dict = {}
                for row in results:
                    key = row['formatted_time']
                    data_dict[key] = row['avg_value'] if row['avg_value'] is not None else 0
                
                current = start
                while current <= end:
                    key = current.strftime('%Y-%m-%d')
                    labels.append(current.strftime('%m/%d'))
                    data.append(data_dict.get(key, 0))
                    current += timedelta(days=1)
                    
            elif grouping == 'weekly':
                # Generate all weeks in the range
                data_dict = {}
                for row in results:
                    key = str(row['time_group'])
                    data_dict[key] = row['avg_value'] if row['avg_value'] is not None else 0
                
                current = start
                while current <= end:
                    year, week, _ = current.isocalendar()
                    yearweek = f"{year}{week:02d}"
                    key = yearweek
                    labels.append(f"Week {week}, {year}")
                    data.append(data_dict.get(key, 0))
                    current += timedelta(weeks=1)
                    
            elif grouping == 'monthly':
                # Generate all months in the range
                data_dict = {}
                for row in results:
                    year = row['year_val']
                    month = row['month_val']
                    key = f"{year}-{month:02d}"
                    data_dict[key] = row['avg_value'] if row['avg_value'] is not None else 0
                
                current = start.replace(day=1)
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                while current <= end:
                    key = f"{current.year}-{current.month:02d}"
                    labels.append(f"{month_names[current.month-1]} {current.year}")
                    data.append(data_dict.get(key, 0))
                    
                    if current.month == 12:
                        current = current.replace(year=current.year + 1, month=1)
                    else:
                        current = current.replace(month=current.month + 1)
                        
            elif grouping == 'yearly':
                # Generate all years in the range
                data_dict = {}
                for row in results:
                    year = row['year_val']
                    key = str(year)
                    data_dict[key] = row['avg_value'] if row['avg_value'] is not None else 0
                
                start_year = start.year
                end_year = end.year
                
                for year in range(start_year, end_year + 1):
                    key = str(year)
                    labels.append(str(year))
                    data.append(data_dict.get(key, 0))
        else:
            # Handle preset ranges normally
            for row in results:
                if grouping == 'weekly':
                    year_week = str(row['time_group'])
                    year = year_week[:4]
                    week = year_week[4:]
                    labels.append(f"Week {week}, {year}")
                elif grouping == 'monthly':
                    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    year = row['year_val']
                    month = row['month_val']
                    labels.append(f"{month_names[month-1]} {year}")
                elif grouping == 'yearly':
                    year = row['year_val']
                    labels.append(str(year))
                else:
                    labels.append(row['formatted_time'])
                data.append(row['avg_value'] if row['avg_value'] is not None else 0)
        
        return jsonify({
            'labels': labels,
            'data': data,
            'parameter': parameter,
            'location': location
        })
        
    except Exception as e:
        print(f"Chart data error: {e}")
        return jsonify({'error': 'Database error'}), 500


@app.route('/dashboard3')
@login_required
def dashboard3():
    user_locations = get_user_locations()
    is_admin = session.get('role') == 'admin'
    
    if is_admin:
        available_locations = AVAILABLE_LOCATIONS
    else:
        # KALAYAAN LOCATIONS - COMMENTED OUT FOR BOTOCAN-ONLY FOCUS
        # kalayaan_locations = ['kalayaan01', 'kalayaan02', 'kalayaan03', 'kalayaan04']
        botocan_locations = ['botocan1', 'botocan2']
        
        # KALAYAAN ACCESS CHECK - COMMENTED OUT FOR BOTOCAN-ONLY FOCUS
        # has_kalayaan_access = any(loc in user_locations for loc in kalayaan_locations)
        has_botocan_access = any(loc in user_locations for loc in botocan_locations)
        
        available_locations = []
        
        for loc in AVAILABLE_LOCATIONS:
            if loc['id'] in user_locations:
                available_locations.append(loc)
            # KALAYAAN ACCESS CHECK - COMMENTED OUT FOR BOTOCAN-ONLY FOCUS
            # elif has_kalayaan_access and loc['id'] in kalayaan_locations:
            #     available_locations.append(loc)
            elif has_botocan_access and loc['id'] in botocan_locations:
                available_locations.append(loc)
    
    return render_template(
        'admin_dashboard3.html',
        is_admin=is_admin,
        available_locations=available_locations,
        user_locations=user_locations
    )


@app.route('/main_dashboard')
@login_required
def main_dashboard():
    user_locations = get_user_locations()
    is_admin = session.get('role') == 'admin'
    
    if is_admin:
        available_locations = AVAILABLE_LOCATIONS
    else:
        # KALAYAAN LOCATIONS - COMMENTED OUT FOR BOTOCAN-ONLY FOCUS
        # kalayaan_locations = ['kalayaan01', 'kalayaan02', 'kalayaan03', 'kalayaan04']
        botocan_locations = ['botocan1', 'botocan2']
        
        # KALAYAAN ACCESS CHECK - COMMENTED OUT FOR BOTOCAN-ONLY FOCUS
        # has_kalayaan_access = any(loc in user_locations for loc in kalayaan_locations)
        has_botocan_access = any(loc in user_locations for loc in botocan_locations)
        
        available_locations = []
        
        for loc in AVAILABLE_LOCATIONS:
            if loc['id'] in user_locations:
                available_locations.append(loc)
            # KALAYAAN ACCESS CHECK - COMMENTED OUT FOR BOTOCAN-ONLY FOCUS
            # elif has_kalayaan_access and loc['id'] in kalayaan_locations:
            #     available_locations.append(loc)
            elif has_botocan_access and loc['id'] in botocan_locations:
                available_locations.append(loc)
    
    return render_template(
        'main_dashboard.html',
        is_admin=is_admin,
        available_locations=available_locations,
        user_locations=user_locations
    )

@app.route('/api/weather-data')
@login_required
def get_weather_data():
    location = request.args.get('location', 'botocan')
    table = request.args.get('table')
    limit = int(request.args.get('limit', 1))
    
    try:
        if table:
            table_name = table
        else:
            if location == 'botocan':
                table_name = 'openmeteo_bhepp_15m'
            # KALAYAAN WEATHER TABLE - COMMENTED OUT FOR BOTOCAN-ONLY FOCUS
            # elif location == 'kalayaan':
            #     table_name = 'openmeteo_kpspp_15m'
            else:
                return jsonify({'error': 'Invalid location'}), 400
        
        cur = mysql.connection.cursor()
        
        query = f"""
            SELECT ts as timestamp, t as temperature, rh as humidity, 
                   ws as wind_speed, sr as solar_radiation, rr as rainfall, 
                   sp as pressure, wd as wind_direction
            FROM {table_name} 
            ORDER BY ts DESC 
            LIMIT %s
        """
        
        cur.execute(query, (limit,))
        results = cur.fetchall()
        cur.close()
        
        return jsonify({
            'data': results,
            'location': location,
            'count': len(results)
        })
            
    except Exception as e:
        print(f"Weather data error: {e}")
        return jsonify({'error': str(e)}), 500

def get_param_name(param):
    """Convert parameter code to readable name"""
    param_map = {
        't': 'temperature',
        'rh': 'humidity', 
        'ws': 'wind_speed',
        'sr': 'solar_radiation',
        'rr': 'rainfall',
        'sp': 'pressure',
        'wd': 'wind_direction'
    }
    return param_map.get(param, param)


@app.route('/api/dashboard-summary')
@login_required
def get_dashboard_summary():
    # CHANGED DEFAULT FROM KALAYAAN01 TO BOTOCAN FOR BOTOCAN-ONLY FOCUS
    location = request.args.get('location', 'botocan')
    
    if not can_access_location(location):
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        # COMMENTED OUT 'ALL' OPTION FOR BOTOCAN-ONLY FOCUS
        # if location == 'all':
        #     data = get_aggregated_data()
        if location == 'botocan_all':
            data = get_botocan_aggregated_data()
        else:
            data = get_real_data_from_db(location)
        
        if data:
            return jsonify({
                'total_power': round(data.get('mw', 0), 1),
                'water_level': round(data.get('upper_water1', 0), 1),
                'temperature': round(data.get('temp_water', 0), 1)
            })
        else:
            return jsonify({'error': 'No data available'}), 404
        
    except Exception as e:
        print(f"Dashboard summary error: {e}")
        return jsonify({'error': 'Database error'}), 500


@app.route('/api/system-stats')
@login_required
def get_system_stats():
    try:
        stats = {
            'active_stations': 2 if session.get('role') == 'admin' else 1,
            'active_alerts': 0,
            'forecast_accuracy': 94.2,
            'last_update': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(stats)
        
    except Exception as e:
        print(f"System stats error: {e}")
        return jsonify({'error': 'Unable to fetch system statistics'}), 500


@app.route('/admin_user_management')
@login_required
def admin_user_management():
    if session.get('role') != 'admin':
        return redirect(url_for('dashboard'))
    
    return render_template('admin_user_management.html')


@app.route('/api/users')
@login_required
def get_users():
    if session.get('role') != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT username, role, locations FROM system_users WHERE archive = 'no'")
        users = cur.fetchall()
        cur.close()
        return jsonify(users)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/users/archived')
@login_required
def get_archived_users():
    if session.get('role') != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT username, role, locations FROM system_users WHERE archive = 'yes'")
        users = cur.fetchall()
        cur.close()
        return jsonify(users)
    except Exception as e:
        return jsonify({'error': str(e)}), 500




# Store OTP codes temporarily (in production, use Redis or database)
#otp_storage = {}

#def xgenerate_otp():
#    """Generate a 6-digit OTP"""
#    return str(random.randint(100000, 999999))

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_password_strength(password):
    """Validate password meets security requirements"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    if not any(c.isupper() for c in password):
        return False, "Password must contain at least one uppercase letter"
    
    if not any(c.islower() for c in password):
        return False, "Password must contain at least one lowercase letter"
    
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    
    if not any(c in '!@#$%^&*(),.?":{}|<>' for c in password):
        return False, "Password must contain at least one special character"
    
    return True, "Password is valid"

#def xsend_email_otp(email, otp):
#    """Send OTP via email"""
#    try:
#        msg = MIMEMultipart('alternative')
#        msg['From'] = f"{SMTP_SENDER_NAME} <{SMTP_USERNAME}>"
#        msg['To'] = email
#        msg['Subject'] = "CBK System - Email Verification Code"
#        
#        html_body = OTP_EMAIL_TEMPLATE.format(otp=otp)
#        msg.attach(MIMEText(html_body, 'html'))
#        
#        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
#        server.starttls()
#        server.login(SMTP_USERNAME, SMTP_PASSWORD)
#        text = msg.as_string()
#        server.sendmail(SMTP_USERNAME, email, text)
#        server.quit()
#        
#        return True
#    except Exception as e:
#        print(f"Email sending error: {e}")
#        return False

#def xsend_welcome_email(email, role, locations, login_url):
#    """Send welcome email to new user"""
#    try:
#        msg = MIMEMultipart()
#        msg['From'] = SMTP_USERNAME
#        msg['To'] = email
#        msg['Subject'] = "Welcome to CBK System"
#        
#        body = WELCOME_EMAIL_TEMPLATE.format(
#            email=email,
#            role=role,
#            locations=locations,
#            login_url=login_url
#        )
#        msg.attach(MIMEText(body, 'plain'))
#        
#        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
#        server.starttls()
#        server.login(SMTP_USERNAME, SMTP_PASSWORD)
#        text = msg.as_string()
#        server.sendmail(SMTP_USERNAME, email, text)
#        server.quit()
#        
#        return True
#    except Exception as e:
#        print(f"Welcome email sending error: {e}")
#        return False

#@app.route('/api/send-otp', methods=['POST'])
#@login_required
#def xsend_otp():
#    if session.get('role') != 'admin':
#        return jsonify({'error': 'Access denied'}), 403
#    
#    try:
#        # Get admin's email from database
#        cur = mysql.connection.cursor()
#        cur.execute("SELECT username FROM system_users WHERE username = %s", (session['user_id'],))
#        user = cur.fetchone()
#        cur.close()
#        
#        if not user:
#            return jsonify({'error': 'User not found'}), 400
#        
#        admin_email = user['username']  # username is the email
#        
#        # Generate OTP
#        otp = generate_otp()
#        
#        # Store OTP with expiration (10 minutes)
#        otp_storage[session['user_id']] = {
#            'otp': otp,
#            'email': admin_email,
#            'expires': time.time() + 600  # 10 minutes
#        }
#        
#        # Send OTP via email
#        if send_email_otp(admin_email, otp):
#            return jsonify({'success': True, 'message': f'Verification code sent to {admin_email}'})
#        else:
#            return jsonify({'error': 'Failed to send verification code'}), 500
#            
#    except Exception as e:
#        print(f"Send OTP error: {e}")
#        return jsonify({'error': 'Database error occurred'}), 500
#
#@app.route('/api/verify-otp', methods=['POST'])
#@login_required
#def xverify_otp():
#    if session.get('role') != 'admin':
#        return jsonify({'error': 'Access denied'}), 403
#    
#    data = request.get_json()
#    otp = data.get('otp')
#    
#    if not otp:
#        return jsonify({'error': 'OTP is required'}), 400
#    
#    # Check if OTP exists and is valid
#    stored_otp = otp_storage.get(session['user_id'])
#    
#    if not stored_otp:
#        return jsonify({'error': 'No verification code found'}), 400
#    
#    if time.time() > stored_otp['expires']:
#        del otp_storage[session['user_id']]
#        return jsonify({'error': 'Verification code has expired'}), 400
#    
#    if otp != stored_otp['otp']:
#        return jsonify({'error': 'Invalid verification code'}), 400
#    
#    # OTP is valid - remove from storage
#    del otp_storage[session['user_id']]
#    
#    return jsonify({'success': True})

@app.route('/api/users', methods=['POST'])
@login_required
def create_user():
    if session.get('role') != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    role = data.get('role', 'user')
    locations = data.get('locations', '')
    
    # Validate input
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
    
    # Validate email format
    import re
    email_regex = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
    if not re.match(email_regex, email):
        return jsonify({'error': 'Invalid email format'}), 400
    
    # Validate password strength
    is_valid, message = validate_password_strength(password)
    if not is_valid:
        return jsonify({'error': message}), 400
    
    try:
        cur = mysql.connection.cursor()
        
        # Check if user already exists
        cur.execute("SELECT username FROM system_users WHERE username = %s", (email,))
        if cur.fetchone():
            cur.close()
            return jsonify({'error': 'User with this email already exists'}), 400
        
        # Insert new user
        cur.execute(
            "INSERT INTO system_users (username, password, role, locations) VALUES (%s, %s, %s, %s)",
            (email, password, role, locations)
        )
        
        mysql.connection.commit()
        cur.close()
        
#        # Send welcome email
#        login_url = request.url_root
#        location_names = locations.replace(',', ', ') if locations else 'None'
#        send_welcome_email(email, role, location_names, login_url)
#        
#        return jsonify({'success': True, 'message': 'User created successfully and welcome email sent'})
    
    except Exception as e:
        print(f"Create user error: {e}")
        return jsonify({'error': 'Database error occurred'}), 500

@app.route('/api/users/<username>')
@login_required
def get_user(username):
    if session.get('role') != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        cur = mysql.connection.cursor()
        cur.execute("SELECT username, role, locations FROM system_users WHERE username = %s", (username,))
        user = cur.fetchone()
        cur.close()
        
        if user:
            return jsonify(user)
        else:
            return jsonify({'error': 'User not found'}), 404
    
    except Exception as e:
        print(f"Get user error: {e}")
        return jsonify({'error': 'Database error occurred'}), 500

@app.route('/api/users/update', methods=['POST'])
@login_required
def update_user():
    if session.get('role') != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.get_json()
    old_username = data.get('oldUsername')
    new_email = data.get('newEmail')
    role = data.get('role')
    locations = data.get('locations', '')
    
    # Validate input
    if not old_username or not new_email:
        return jsonify({'error': 'Username and email are required'}), 400
    
    # Validate email format
    import re
    email_regex = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
    if not re.match(email_regex, new_email):
        return jsonify({'error': 'Invalid email format'}), 400
    
    try:
        cur = mysql.connection.cursor()
        
        # Check if new email already exists (if different from current)
        if old_username != new_email:
            cur.execute("SELECT username FROM system_users WHERE username = %s", (new_email,))
            if cur.fetchone():
                cur.close()
                return jsonify({'error': 'User with this email already exists'}), 400
        
        # Update user
        cur.execute(
            "UPDATE system_users SET username = %s, role = %s, locations = %s WHERE username = %s",
            (new_email, role, locations, old_username)
        )
        
        mysql.connection.commit()
        cur.close()
        
        return jsonify({'success': True, 'message': 'User updated successfully'})
    
    except Exception as e:
        print(f"Update user error: {e}")
        return jsonify({'error': 'Database error occurred'}), 500

@app.route('/api/admin/change-password', methods=['POST'])
@login_required
def admin_change_password():
    if session.get('role') != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.get_json()
    old_password = data.get('old_password')
    new_password = data.get('new_password')
    
    # Validate input
    if not old_password or not new_password:
        return jsonify({'error': 'Old and new passwords are required'}), 400
    
    # Validate password strength
    is_valid, message = validate_password_strength(new_password)
    if not is_valid:
        return jsonify({'error': message}), 400
    
    try:
        cur = mysql.connection.cursor()
        
        # Verify old password
        cur.execute("SELECT password FROM system_users WHERE username = %s", (session['user_id'],))
        user = cur.fetchone()
        
        if not user or user['password'] != old_password:
            cur.close()
            return jsonify({'error': 'Current password is incorrect'}), 400
        
        # Update password
        cur.execute(
            "UPDATE system_users SET password = %s WHERE username = %s",
            (new_password, session['user_id'])
        )
        
        mysql.connection.commit()
        cur.close()
        
        return jsonify({'success': True, 'message': 'Password changed successfully'})
    
    except Exception as e:
        print(f"Change password error: {e}")
        return jsonify({'error': 'Database error occurred'}), 500

@app.route('/api/users/reset-password', methods=['POST'])
@login_required
def reset_user_password():
    if session.get('role') != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    
    # Validate input
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    
    # Validate password strength
    is_valid, message = validate_password_strength(password)
    if not is_valid:
        return jsonify({'error': message}), 400
    
    try:
        cur = mysql.connection.cursor()
        
        # Update password
        cur.execute(
            "UPDATE system_users SET password = %s WHERE username = %s",
            (password, username)
        )
        
        mysql.connection.commit()
        cur.close()
        
        return jsonify({'success': True, 'message': 'Password reset successfully'})
    
    except Exception as e:
        print(f"Reset password error: {e}")
        return jsonify({'error': 'Database error occurred'}), 500

@app.route('/api/users/archive', methods=['POST'])
@login_required
def archive_user():
    if session.get('role') != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.get_json()
    username = data.get('username')
    
    # Prevent admin from archiving themselves
    if username == session.get('user_id'):
        return jsonify({'error': 'Cannot archive your own account'}), 400
    
    try:
        cur = mysql.connection.cursor()
        
        # Check if user exists and is not already archived
        cur.execute("SELECT username FROM system_users WHERE username = %s AND archive = 'no'", (username,))
        if not cur.fetchone():
            cur.close()
            return jsonify({'error': 'User not found or already archived'}), 404
        
        # Archive user
        cur.execute("UPDATE system_users SET archive = 'yes' WHERE username = %s", (username,))
        
        mysql.connection.commit()
        cur.close()
        
        return jsonify({'success': True, 'message': 'User archived successfully'})
    
    except Exception as e:
        print(f"Archive user error: {e}")
        return jsonify({'error': 'Database error occurred'}), 500

@app.route('/api/users/restore', methods=['POST'])
@login_required
def restore_user():
    if session.get('role') != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.get_json()
    username = data.get('username')
    
    try:
        cur = mysql.connection.cursor()
        
        # Check if user exists and is archived
        cur.execute("SELECT username FROM system_users WHERE username = %s AND archive = 'yes'", (username,))
        if not cur.fetchone():
            cur.close()
            return jsonify({'error': 'User not found or not archived'}), 404
        
        # Restore user
        cur.execute("UPDATE system_users SET archive = 'no' WHERE username = %s", (username,))
        
        mysql.connection.commit()
        cur.close()
        
        return jsonify({'success': True, 'message': 'User restored successfully'})
    
    except Exception as e:
        print(f"Restore user error: {e}")
        return jsonify({'error': 'Database error occurred'}), 500

@app.route('/api/users/permanent-delete', methods=['POST'])
@login_required
def permanent_delete_user():
    if session.get('role') != 'admin':
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.get_json()
    username = data.get('username')
    
    # Prevent admin from permanently deleting themselves
    if username == session.get('user_id'):
        return jsonify({'error': 'Cannot delete your own account'}), 400
    
    try:
        cur = mysql.connection.cursor()
        
        # Check if user exists and is archived
        cur.execute("SELECT username FROM system_users WHERE username = %s AND archive = 'yes'", (username,))
        if not cur.fetchone():
            cur.close()
            return jsonify({'error': 'User not found or not archived'}), 404
        
        # Permanently delete user
        cur.execute("DELETE FROM system_users WHERE username = %s", (username,))
        
        mysql.connection.commit()
        cur.close()
        
        return jsonify({'success': True, 'message': 'User permanently deleted'})
    
    except Exception as e:
        print(f"Permanent delete user error: {e}")
        return jsonify({'error': 'Database error occurred'}), 500

@app.route('/forecasting')
@login_required
def forecasting():
    return render_template('forecasting.html')

@app.route('/api/train-model', methods=['POST'])
@login_required
def api_train_model():
    try:
        from forecast_api import train_model
        
        cur = mysql.connection.cursor()
        cur.execute("SELECT ts, mw, upper_water, rr FROM b01_parameters_hourly ORDER BY ts")
        b01_data = cur.fetchall()
        cur.execute("SELECT ts, mw, upper_water, rr FROM b02_parameters_hourly ORDER BY ts")
        b02_data = cur.fetchall()
        cur.close()
        
        df_b01 = pd.DataFrame(b01_data)
        df_b02 = pd.DataFrame(b02_data)
        df = pd.merge(df_b01, df_b02, on='ts', suffixes=('_b01', '_b02'), how='outer').sort_values('ts')
        df['date'] = pd.to_datetime(df['ts'])
        df['b12_mw'] = df['mw_b01'] + df['mw_b02']
        df['dam_el'] = df['upper_water_b01']
        df['rainfall'] = df[['rr_b01', 'rr_b02']].mean(axis=1)
        df['dam_el_delta'] = df['dam_el'].diff()
        df = df[['date', 'b12_mw', 'dam_el', 'dam_el_delta', 'rainfall']]
        
        result = train_model(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/run-prediction', methods=['POST'])
@login_required
def api_run_prediction():
    try:
        from forecast_api import run_prediction
        
        cur = mysql.connection.cursor()
        cur.execute("SELECT ts, mw, upper_water, rr FROM b01_parameters_hourly ORDER BY ts")
        b01_data = cur.fetchall()
        cur.execute("SELECT ts, mw, upper_water, rr FROM b02_parameters_hourly ORDER BY ts")
        b02_data = cur.fetchall()
        cur.close()
        
        df_b01 = pd.DataFrame(b01_data)
        df_b02 = pd.DataFrame(b02_data)
        df = pd.merge(df_b01, df_b02, on='ts', suffixes=('_b01', '_b02'), how='outer').sort_values('ts')
        df['date'] = pd.to_datetime(df['ts'])
        df['b12_mw'] = df['mw_b01'] + df['mw_b02']
        df['dam_el'] = df['upper_water_b01']
        df['rainfall'] = df[['rr_b01', 'rr_b02']].mean(axis=1)
        df['dam_el_delta'] = df['dam_el'].diff()
        df = df[['date', 'b12_mw', 'dam_el', 'dam_el_delta', 'rainfall']]
        
        result = run_prediction(df)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/forecast-data')
@login_required
def api_forecast_data():
    try:
        from forecast_api import get_forecast_data
        forecast_data = get_forecast_data()
        
        # Fetch rainfall from database
        cur = mysql.connection.cursor()
        cur.execute("""
            SELECT ts, rr 
            FROM openmeteo_bhepp_7d 
            WHERE ts >= NOW() 
            ORDER BY ts ASC 
            LIMIT 168
        """)
        rainfall_data = cur.fetchall()
        cur.close()
        
        # Update rainfall in forecast data
        if rainfall_data and 'rain_mm' in forecast_data:
            forecast_data['rain_mm'] = [row['rr'] if row['rr'] is not None else 0 for row in rainfall_data]
        
        return jsonify(forecast_data)
    except Exception as e:
        return jsonify({"error": str(e)})

# ============================================================================
# APPLICATION STARTUP
# ============================================================================

# Run the Flask application when script is executed directly
if __name__ == '__main__':
    # host='0.0.0.0' allows access from any IP address (not just localhost)
    # port=5000 sets the port number
    # debug=True enables auto-reload and detailed error messages (disable in production)
    app.run(host='0.0.0.0', port=80, debug=True)
