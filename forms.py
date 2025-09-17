from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, FloatField, DateField, DateTimeField, TextAreaField, SelectField, SubmitField, PasswordField, BooleanField, HiddenField
from wtforms.validators import DataRequired, NumberRange, Optional, Length, Email, EqualTo, ValidationError, URL
from datetime import datetime
from models.sqlalchemy_models import User

class RaceForm(FlaskForm):
    # Basic Race Information
    name = StringField('Race Name', validators=[Optional(), Length(min=2, max=100)])
    date = DateTimeField('Race Date & Time', validators=[Optional()], default=datetime.now, format='%Y-%m-%dT%H:%M')
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

    # Horse Assignment
    assigned_horses = SelectField('Add Horse to Race', coerce=str, validators=[Optional()])
    horse_post_position = IntegerField('Horse Post Position', validators=[Optional(), NumberRange(min=1, max=20)])
    horse_morning_line_odds = StringField('Horse Morning Line Odds', validators=[Optional(), Length(max=10)])

    submit = SubmitField('Create Race')

class HorseForm(FlaskForm):
    # Basic Information
    name = StringField('Horse Name', validators=[Optional(), Length(min=2, max=100)])
    age = IntegerField('Age', validators=[Optional(), NumberRange(min=2, max=20)])
    sex = SelectField('Sex', 
                     choices=[
                         ('', 'Select Sex'),
                         ('colt', 'Colt'),
                         ('filly', 'Filly'),
                         ('gelding', 'Gelding'),
                         ('mare', 'Mare'),
                         ('stallion', 'Stallion')
                     ],
                     validators=[Optional()])
    color = SelectField('Color',
                       choices=[
                           ('', 'Select Color'),
                           ('bay', 'Bay'),
                           ('brown', 'Brown'),
                           ('chestnut', 'Chestnut'),
                           ('black', 'Black'),
                           ('gray', 'Gray'),
                           ('roan', 'Roan'),
                           ('palomino', 'Palomino'),
                           ('pinto', 'Pinto'),
                           ('dun', 'Dun'),
                           ('buckskin', 'Buckskin'),
                           ('cremello', 'Cremello'),
                           ('perlino', 'Perlino'),
                           ('white', 'White')
                       ],
                       validators=[Optional()])
    breed = SelectField('Breed',
                       choices=[
                           ('', 'Select Breed'),
                           ('thoroughbred', 'Thoroughbred'),
                           ('quarter_horse', 'Quarter Horse'),
                           ('standardbred', 'Standardbred'),
                           ('arabian', 'Arabian'),
                           ('appaloosa', 'Appaloosa'),
                           ('paint', 'Paint'),
                           ('tennessee_walker', 'Tennessee Walking Horse'),
                           ('morgan', 'Morgan'),
                           ('friesian', 'Friesian'),
                           ('clydesdale', 'Clydesdale'),
                           ('percheron', 'Percheron'),
                           ('mustang', 'Mustang'),
                           ('warmblood', 'Warmblood'),
                           ('other', 'Other')
                       ],
                       validators=[Optional()])
    
    # Connections
    owner = StringField('Owner', validators=[Optional(), Length(max=100)])
    trainer = StringField('Trainer', validators=[Optional(), Length(max=100)])
    jockey = StringField('Jockey', validators=[Optional(), Length(max=100)])
    
    # Pedigree Information
    sire = StringField('Sire', validators=[Optional(), Length(max=100)])
    dam = StringField('Dam', validators=[Optional(), Length(max=100)])
    
    # Speed Ratings - Last 3 Races
    speed_rating_race_1 = IntegerField('Speed Rating - Most Recent Race', validators=[Optional(), NumberRange(min=0, max=130)])
    speed_rating_race_2 = IntegerField('Speed Rating - 2nd Most Recent Race', validators=[Optional(), NumberRange(min=0, max=130)])
    speed_rating_race_3 = IntegerField('Speed Rating - 3rd Most Recent Race', validators=[Optional(), NumberRange(min=0, max=130)])
    
    # Highest Speed Rating from Last 10 Races for Today's Distance
    highest_speed_rating_distance = IntegerField('Highest Speed Rating (Last 10 Races at Today\'s Distance)', validators=[Optional(), NumberRange(min=0, max=130)])
    
    # Beaten Lengths in Previous 3 Races
    beaten_lengths_race_1 = FloatField('Beaten Lengths - Most Recent Race', validators=[Optional(), NumberRange(min=0, max=50)])
    beaten_lengths_race_2 = FloatField('Beaten Lengths - 2nd Most Recent Race', validators=[Optional(), NumberRange(min=0, max=50)])
    beaten_lengths_race_3 = FloatField('Beaten Lengths - 3rd Most Recent Race', validators=[Optional(), NumberRange(min=0, max=50)])
    
    # Pace-Related Running Style
    running_style = SelectField('Running Style (Pace Position)',
                              choices=[
                                  ('', 'Select Running Style'),
                                  ('pace_leader', 'Pace Leader'),
                                  ('contesting_pace', 'Contesting Pace'),
                                  ('stalking', 'Stalking'),
                                  ('closing', 'Closing')
                              ],
                              validators=[Optional()])
    
    # Pace Analysis
    early_pace_ability = SelectField('Early Pace Ability',
                                   choices=[
                                       ('', 'Select Ability'),
                                       ('excellent', 'Excellent'),
                                       ('good', 'Good'),
                                       ('average', 'Average'),
                                       ('poor', 'Poor')
                                   ],
                                   validators=[Optional()])
    
    late_pace_ability = SelectField('Late Pace Ability (Closing Kick)',
                                  choices=[
                                      ('', 'Select Ability'),
                                      ('excellent', 'Excellent'),
                                      ('good', 'Good'),
                                      ('average', 'Average'),
                                      ('poor', 'Poor')
                                  ],
                                  validators=[Optional()])
    
    pace_versatility = SelectField('Pace Versatility',
                                 choices=[
                                     ('', 'Select Versatility'),
                                     ('very_versatile', 'Very Versatile'),
                                     ('somewhat_versatile', 'Somewhat Versatile'),
                                     ('limited', 'Limited'),
                                     ('one_dimensional', 'One Dimensional')
                                 ],
                                 validators=[Optional()])
    
    # Pedigree - Speed/Stamina Capability
    pedigree_speed_rating = SelectField('Pedigree Speed Rating',
                                      choices=[
                                          ('', 'Select Rating'),
                                          ('excellent', 'Excellent Speed Pedigree'),
                                          ('good', 'Good Speed Pedigree'),
                                          ('average', 'Average Speed Pedigree'),
                                          ('poor', 'Poor Speed Pedigree')
                                      ],
                                      validators=[Optional()])
    
    pedigree_stamina_rating = SelectField('Pedigree Stamina Rating',
                                        choices=[
                                            ('', 'Select Rating'),
                                            ('excellent', 'Excellent Stamina Pedigree'),
                                            ('good', 'Good Stamina Pedigree'),
                                            ('average', 'Average Stamina Pedigree'),
                                            ('poor', 'Poor Stamina Pedigree')
                                        ],
                                        validators=[Optional()])
    
    distance_pedigree_suitability = SelectField('Distance Suitability (Pedigree)',
                                              choices=[
                                                  ('', 'Select Suitability'),
                                                  ('sprint_specialist', 'Sprint Specialist (5f-7f)'),
                                                  ('miler', 'Miler (7f-1m)'),
                                                  ('middle_distance', 'Middle Distance (1m-1.25m)'),
                                                  ('router', 'Router (1.25m+)'),
                                                  ('versatile', 'Versatile (All Distances)')
                                              ],
                                              validators=[Optional()])
    # Trainer/Jockey Performance
    trainer_win_percentage = FloatField('Trainer Win Percentage (%)', validators=[Optional(), NumberRange(min=0, max=100)])
    trainer_win_percentage_distance = FloatField('Trainer Win % at This Distance (%)', validators=[Optional(), NumberRange(min=0, max=100)])
    trainer_win_percentage_surface = FloatField('Trainer Win % on This Surface (%)', validators=[Optional(), NumberRange(min=0, max=100)])
    
    jockey_win_percentage = FloatField('Jockey Win Percentage (%)', validators=[Optional(), NumberRange(min=0, max=100)])
    trainer_jockey_combo_wins = IntegerField('Trainer/Jockey Combo Wins', validators=[Optional(), NumberRange(min=0, max=1000)])

    # Race Assignment
    assigned_races = SelectField('Assign to Race', coerce=str, validators=[Optional()])
    post_position = IntegerField('Post Position', validators=[Optional(), NumberRange(min=1, max=20)])
    morning_line_odds = StringField('Morning Line Odds', validators=[Optional(), Length(max=10)])

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