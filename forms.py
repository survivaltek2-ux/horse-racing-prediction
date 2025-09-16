from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, FloatField, DateField, DateTimeField, TextAreaField, SelectField, SubmitField, PasswordField, BooleanField, HiddenField
from wtforms.validators import DataRequired, NumberRange, Optional, Length, Email, EqualTo, ValidationError, URL
from datetime import datetime
from models.user import User

class RaceForm(FlaskForm):
    # Basic Race Information
    name = StringField('Race Name', validators=[DataRequired(), Length(min=2, max=100)])
    date = DateTimeField('Race Date & Time', validators=[DataRequired()], default=datetime.now)
    location = StringField('Location', validators=[Optional(), Length(max=100)])
    distance = StringField('Distance', validators=[Optional(), Length(max=50)])
    purse = FloatField('Purse Amount', validators=[Optional(), NumberRange(min=0)])
    description = TextAreaField('Description', validators=[Optional(), Length(max=500)])
    
    # Weather Conditions
    temperature = IntegerField('Temperature (°F)', validators=[Optional(), NumberRange(min=-20, max=120)])
    humidity = IntegerField('Humidity (%)', validators=[Optional(), NumberRange(min=0, max=100)])
    wind_speed = IntegerField('Wind Speed (mph)', validators=[Optional(), NumberRange(min=0, max=50)])
    wind_direction = SelectField('Wind Direction',
                               choices=[
                                   ('', 'Select Direction'),
                                   ('N', 'North'),
                                   ('NE', 'Northeast'),
                                   ('E', 'East'),
                                   ('SE', 'Southeast'),
                                   ('S', 'South'),
                                   ('SW', 'Southwest'),
                                   ('W', 'West'),
                                   ('NW', 'Northwest')
                               ],
                               validators=[Optional()])
    weather_description = SelectField('Weather Description',
                                    choices=[
                                        ('', 'Select Weather'),
                                        ('Clear', 'Clear'),
                                        ('Partly Cloudy', 'Partly Cloudy'),
                                        ('Overcast', 'Overcast'),
                                        ('Light Rain', 'Light Rain'),
                                        ('Heavy Rain', 'Heavy Rain'),
                                        ('Fog', 'Fog')
                                    ],
                                    validators=[Optional()])
    visibility = FloatField('Visibility (miles)', validators=[Optional(), NumberRange(min=0, max=20)])
    
    # Track Conditions
    track_condition = SelectField('Track Condition', 
                                choices=[
                                    ('good', 'Good'),
                                    ('firm', 'Firm'),
                                    ('soft', 'Soft'),
                                    ('heavy', 'Heavy'),
                                    ('fast', 'Fast'),
                                    ('muddy', 'Muddy'),
                                    ('sloppy', 'Sloppy')
                                ],
                                default='good')
    surface_type = SelectField('Surface Type',
                             choices=[
                                 ('', 'Select Surface'),
                                 ('Dirt', 'Dirt'),
                                 ('Turf', 'Turf'),
                                 ('Synthetic', 'Synthetic'),
                                 ('All Weather', 'All Weather')
                             ],
                             validators=[Optional()])
    rail_position = StringField('Rail Position', validators=[Optional(), Length(max=50)])
    track_bias = SelectField('Track Bias',
                           choices=[
                               ('', 'Select Bias'),
                               ('None', 'None'),
                               ('Speed Favoring', 'Speed Favoring'),
                               ('Closer Favoring', 'Closer Favoring'),
                               ('Inside Bias', 'Inside Bias'),
                               ('Outside Bias', 'Outside Bias')
                           ],
                           validators=[Optional()])
    track_maintenance = StringField('Track Maintenance Notes', validators=[Optional(), Length(max=200)])
    
    # Field Analysis
    field_size = IntegerField('Field Size', validators=[Optional(), NumberRange(min=2, max=20)])
    field_quality = SelectField('Field Quality',
                              choices=[
                                  ('', 'Select Quality'),
                                  ('Weak', 'Weak'),
                                  ('Average', 'Average'),
                                  ('Strong', 'Strong'),
                                  ('Elite', 'Elite')
                              ],
                              validators=[Optional()])
    pace_scenario = SelectField('Pace Scenario',
                              choices=[
                                  ('', 'Select Pace'),
                                  ('Slow', 'Slow'),
                                  ('Moderate', 'Moderate'),
                                  ('Fast', 'Fast'),
                                  ('Contested', 'Contested')
                              ],
                              validators=[Optional()])
    competitive_balance = SelectField('Competitive Balance',
                                    choices=[
                                        ('', 'Select Balance'),
                                        ('Even', 'Even'),
                                        ('Top Heavy', 'Top Heavy'),
                                        ('Wide Open', 'Wide Open'),
                                        ('Dominant Favorite', 'Dominant Favorite')
                                    ],
                                    validators=[Optional()])
    speed_figures_range = StringField('Speed Figures Range', validators=[Optional(), Length(max=20)])
    
    # Betting Information
    total_pool = FloatField('Total Pool ($)', validators=[Optional(), NumberRange(min=0)])
    win_pool = FloatField('Win Pool ($)', validators=[Optional(), NumberRange(min=0)])
    exacta_pool = FloatField('Exacta Pool ($)', validators=[Optional(), NumberRange(min=0)])
    trifecta_pool = FloatField('Trifecta Pool ($)', validators=[Optional(), NumberRange(min=0)])
    superfecta_pool = FloatField('Superfecta Pool ($)', validators=[Optional(), NumberRange(min=0)])
    morning_line_favorite = StringField('Morning Line Favorite Odds', validators=[Optional(), Length(max=10)])
    
    # Race Conditions
    age_restrictions = StringField('Age Restrictions', validators=[Optional(), Length(max=50)])
    sex_restrictions = SelectField('Sex Restrictions',
                                 choices=[
                                     ('', 'No Restrictions'),
                                     ('Fillies and Mares', 'Fillies and Mares'),
                                     ('Colts and Geldings', 'Colts and Geldings'),
                                     ('Fillies Only', 'Fillies Only'),
                                     ('Mares Only', 'Mares Only')
                                 ],
                                 validators=[Optional()])
    weight_conditions = StringField('Weight Conditions', validators=[Optional(), Length(max=100)])
    claiming_price = FloatField('Claiming Price ($)', validators=[Optional(), NumberRange(min=0)])
    race_grade = SelectField('Race Grade',
                           choices=[
                               ('', 'Select Grade'),
                               ('Grade 1', 'Grade 1'),
                               ('Grade 2', 'Grade 2'),
                               ('Grade 3', 'Grade 3'),
                               ('Listed', 'Listed'),
                               ('Allowance', 'Allowance'),
                               ('Claiming', 'Claiming'),
                               ('Maiden', 'Maiden')
                           ],
                           validators=[Optional()])
    
    # Historical Data
    track_record = StringField('Track Record Time', validators=[Optional(), Length(max=20)])
    average_winning_time = StringField('Average Winning Time', validators=[Optional(), Length(max=20)])
    course_record_holder = StringField('Course Record Holder', validators=[Optional(), Length(max=100)])
    similar_race_results = TextAreaField('Similar Race Results', validators=[Optional(), Length(max=500)])
    trainer_jockey_stats = TextAreaField('Trainer/Jockey Stats', validators=[Optional(), Length(max=500)])
    
    # Media Coverage
    tv_coverage = SelectField('TV Coverage',
                            choices=[
                                ('', 'Select Coverage'),
                                ('None', 'None'),
                                ('Local', 'Local'),
                                ('National', 'National'),
                                ('International', 'International')
                            ],
                            validators=[Optional()])
    streaming_available = SelectField('Streaming Available',
                                    choices=[
                                        ('', 'Select Option'),
                                        ('Yes', 'Yes'),
                                        ('No', 'No')
                                    ],
                                    validators=[Optional()])
    featured_race = SelectField('Featured Race',
                              choices=[
                                  ('', 'Select Option'),
                                  ('Yes', 'Yes'),
                                  ('No', 'No')
                              ],
                              validators=[Optional()])
    
    submit = SubmitField('Create Race')

class HorseForm(FlaskForm):
    # Basic Information
    name = StringField('Horse Name', validators=[DataRequired(), Length(min=2, max=100)])
    age = IntegerField('Age', validators=[DataRequired(), NumberRange(min=2, max=20)])
    breed = StringField('Breed', validators=[Optional(), Length(max=50)])
    color = StringField('Color', validators=[Optional(), Length(max=30)])
    height = StringField('Height (hands)', validators=[Optional(), Length(max=10)])
    markings = TextAreaField('Markings', validators=[Optional(), Length(max=200)])
    
    # Pedigree Information
    sire = StringField('Sire', validators=[Optional(), Length(max=100)])
    dam = StringField('Dam', validators=[Optional(), Length(max=100)])
    pedigree_rating = IntegerField('Pedigree Rating (1-10)', validators=[Optional(), NumberRange(min=1, max=10)])
    bloodline_notes = TextAreaField('Bloodline Notes', validators=[Optional(), Length(max=500)])
    
    # Connections
    jockey = StringField('Jockey', validators=[Optional(), Length(max=100)])
    trainer = StringField('Trainer', validators=[Optional(), Length(max=100)])
    owner = StringField('Owner', validators=[Optional(), Length(max=100)])
    breeder = StringField('Breeder', validators=[Optional(), Length(max=100)])
    
    # Physical Attributes
    weight = FloatField('Weight (lbs)', validators=[Optional(), NumberRange(min=800, max=1400)])
    body_condition = SelectField('Body Condition',
                               choices=[
                                   ('', 'Select Condition'),
                                   ('poor', 'Poor'),
                                   ('fair', 'Fair'),
                                   ('good', 'Good'),
                                   ('excellent', 'Excellent')
                               ],
                               validators=[Optional()])
    muscle_tone = SelectField('Muscle Tone',
                            choices=[
                                ('', 'Select Tone'),
                                ('poor', 'Poor'),
                                ('average', 'Average'),
                                ('good', 'Good'),
                                ('excellent', 'Excellent')
                            ],
                            validators=[Optional()])
    
    # Performance Analytics
    speed_rating = IntegerField('Speed Rating (0-120)', validators=[Optional(), NumberRange(min=0, max=120)])
    class_rating = IntegerField('Class Rating (1-10)', validators=[Optional(), NumberRange(min=1, max=10)])
    distance_preference = SelectField('Distance Preference', 
                                    choices=[
                                        ('', 'Select Distance'),
                                        ('5f-7f', '5f-7f (Sprint)'),
                                        ('6f-1m', '6f-1m (Middle)'),
                                        ('7f-1.25m', '7f-1.25m (Mile)'),
                                        ('1m-1.5m', '1m-1.5m (Route)'),
                                        ('1m-1.75m', '1m-1.75m (Long Route)'),
                                        ('1.5m+', '1.5m+ (Marathon)')
                                    ],
                                    validators=[Optional()])
    surface_preference = SelectField('Surface Preference',
                                   choices=[
                                       ('', 'Select Surface'),
                                       ('dirt', 'Dirt'),
                                       ('turf', 'Turf'),
                                       ('synthetic', 'Synthetic')
                                   ],
                                   validators=[Optional()])
    track_bias_rating = IntegerField('Track Bias Rating (1-10)', validators=[Optional(), NumberRange(min=1, max=10)])
    best_beyer_speed = IntegerField('Best Beyer Speed Figure', validators=[Optional(), NumberRange(min=0, max=130)])
    avg_speed_last_3 = IntegerField('Average Speed Last 3 Races', validators=[Optional(), NumberRange(min=0, max=130)])
    
    # Training & Fitness
    days_since_last_race = IntegerField('Days Since Last Race', validators=[Optional(), NumberRange(min=0, max=365)])
    fitness_level = IntegerField('Fitness Level (1-10)', validators=[Optional(), NumberRange(min=1, max=10)])
    training_intensity = SelectField('Training Intensity',
                                   choices=[
                                       ('', 'Select Intensity'),
                                       ('light', 'Light'),
                                       ('moderate', 'Moderate'),
                                       ('heavy', 'Heavy'),
                                       ('intense', 'Intense')
                                   ],
                                   validators=[Optional()])
    workout_times = TextAreaField('Recent Workout Times (JSON)', validators=[Optional(), Length(max=1000)])
    injury_history = TextAreaField('Injury History (JSON)', validators=[Optional(), Length(max=1000)])
    recovery_status = SelectField('Recovery Status',
                                choices=[
                                    ('', 'Select Status'),
                                    ('fully_recovered', 'Fully Recovered'),
                                    ('recovering', 'Recovering'),
                                    ('minor_issue', 'Minor Issue'),
                                    ('major_concern', 'Major Concern')
                                ],
                                validators=[Optional()])
    
    # Behavioral & Racing Style
    temperament = SelectField('Temperament',
                            choices=[
                                ('', 'Select Temperament'),
                                ('calm', 'Calm'),
                                ('nervous', 'Nervous'),
                                ('aggressive', 'Aggressive'),
                                ('lazy', 'Lazy'),
                                ('eager', 'Eager')
                            ],
                            validators=[Optional()])
    running_style = SelectField('Running Style',
                              choices=[
                                  ('', 'Select Style'),
                                  ('front_runner', 'Front Runner'),
                                  ('presser', 'Presser'),
                                  ('stalker', 'Stalker'),
                                  ('closer', 'Closer')
                              ],
                              validators=[Optional()])
    gate_behavior = SelectField('Gate Behavior',
                              choices=[
                                  ('', 'Select Behavior'),
                                  ('excellent', 'Excellent'),
                                  ('good', 'Good'),
                                  ('average', 'Average'),
                                  ('poor', 'Poor')
                              ],
                              validators=[Optional()])
    
    # Financial Information
    purchase_price = FloatField('Purchase Price ($)', validators=[Optional(), NumberRange(min=0)])
    current_value = FloatField('Current Estimated Value ($)', validators=[Optional(), NumberRange(min=0)])
    earnings_to_date = FloatField('Earnings to Date ($)', validators=[Optional(), NumberRange(min=0)])
    insurance_value = FloatField('Insurance Value ($)', validators=[Optional(), NumberRange(min=0)])
    
    submit = SubmitField('Add Horse')

class PredictionForm(FlaskForm):
    algorithm = SelectField('Prediction Algorithm',
                          choices=[
                              ('heuristic', 'Heuristic Analysis'),
                              ('machine_learning', 'Machine Learning'),
                              ('statistical', 'Statistical Model')
                          ],
                          default='heuristic',
                          validators=[DataRequired()])
    submit = SubmitField('Generate Prediction')

class RaceResultForm(FlaskForm):
    winner_id = SelectField('Winner', coerce=str, validators=[DataRequired()])
    second_place_id = SelectField('Second Place', coerce=str, validators=[Optional()])
    third_place_id = SelectField('Third Place', coerce=str, validators=[Optional()])
    winning_time = StringField('Winning Time (mm:ss.ss)', validators=[Optional(), Length(max=20)])
    notes = TextAreaField('Race Notes', validators=[Optional(), Length(max=500)])
    submit = SubmitField('Record Results')

class AddHorseToRaceForm(FlaskForm):
    horse_id = SelectField('Select Horse', coerce=str, validators=[DataRequired()])
    post_position = IntegerField('Post Position', validators=[Optional(), NumberRange(min=1, max=20)])
    morning_line_odds = StringField('Morning Line Odds', validators=[Optional(), Length(max=10)])
    submit = SubmitField('Add Horse to Race')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=20)])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    password2 = PasswordField('Confirm Password', 
                             validators=[DataRequired(), EqualTo('password', message='Passwords must match')])
    submit = SubmitField('Register')
    
    def validate_username(self, username):
        user = User.get_by_username(username.data)
        if user:
            raise ValidationError('Username already exists. Please choose a different one.')
    
    def validate_email(self, email):
        user = User.get_by_email(email.data)
        if user:
            raise ValidationError('Email already registered. Please choose a different one.')

class UserManagementForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=20)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    role = SelectField('Role', choices=[('user', 'User'), ('admin', 'Administrator')], default='user')
    is_active = BooleanField('Active', default=True)
    password = PasswordField('Password (leave blank to keep current)', validators=[Optional(), Length(min=6)])
    submit = SubmitField('Save User')

class ChangePasswordForm(FlaskForm):
    current_password = PasswordField('Current Password', validators=[DataRequired()])
    new_password = PasswordField('New Password', validators=[DataRequired(), Length(min=6)])
    new_password2 = PasswordField('Confirm New Password', 
                                 validators=[DataRequired(), EqualTo('new_password', message='Passwords must match')])
    submit = SubmitField('Change Password')
    
    def validate_password(self, field):
        if len(field.data) < 6:
            raise ValidationError('Password must be at least 6 characters long.')


class APICredentialsForm(FlaskForm):
    """Form for managing API credentials"""
    
    # Predefined API providers
    PROVIDER_CHOICES = [
        ('', 'Select an API Provider'),
        ('theracingapi', 'The Racing API - Comprehensive UK, Ireland & USA horse racing data'),
        ('odds_api', 'The Odds API - Sports betting odds API'),
        ('rapid_api', 'RapidAPI Horse Racing - Horse racing data via RapidAPI'),
        ('sample', 'Sample Racing API - Example external API'),
        ('mock', 'Mock Racing API - Test API for development'),
        ('custom', 'Custom Provider - Enter your own provider details')
    ]
    
    provider = SelectField('API Provider', choices=PROVIDER_CHOICES, validators=[DataRequired()])
    custom_provider_name = StringField('Custom Provider Name', validators=[Optional(), Length(min=2, max=50)])
    api_key = StringField('API Key', validators=[Optional(), Length(min=5, max=200)])
    api_secret = PasswordField('API Secret', validators=[Optional(), Length(max=200)])
    username = StringField('Username', validators=[Optional(), Length(min=2, max=100)])
    password = PasswordField('Password', validators=[Optional(), Length(min=3, max=200)])
    base_url = StringField('Base URL', validators=[Optional(), URL(), Length(max=200)])
    description = TextAreaField('Description', validators=[Optional(), Length(max=500)])
    is_active = BooleanField('Active', default=True)
    submit = SubmitField('Save Credentials')
    
    def validate_provider(self, field):
        # Check for valid provider name format
        if not field.data.replace('_', '').replace('-', '').isalnum():
            raise ValidationError('Provider name can only contain letters, numbers, hyphens, and underscores.')
    
    def validate(self, extra_validators=None):
        """Custom validation to ensure either API key or username/password is provided"""
        if not super().validate(extra_validators):
            return False
        
        # Check if either API key or username/password combination is provided
        has_api_key = bool(self.api_key.data and self.api_key.data.strip())
        has_username_password = bool(self.username.data and self.username.data.strip() and 
                                   self.password.data and self.password.data.strip())
        
        if not has_api_key and not has_username_password:
            self.api_key.errors.append('Either API Key or Username/Password combination must be provided.')
            self.username.errors.append('Either API Key or Username/Password combination must be provided.')
            return False
        
        return True


class APICredentialsTestForm(FlaskForm):
    """Form for testing API credentials"""
    credential_id = HiddenField('Credential ID', validators=[DataRequired()])
    test_endpoint = StringField('Test Endpoint', validators=[Optional(), Length(max=200)])
    submit = SubmitField('Test Connection')