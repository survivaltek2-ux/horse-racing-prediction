from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, FloatField, DateField, DateTimeField, TextAreaField, SelectField, SubmitField, PasswordField, BooleanField, HiddenField
from wtforms.validators import DataRequired, NumberRange, Optional, Length, Email, EqualTo, ValidationError, URL
from datetime import datetime
from models.user import User

class RaceForm(FlaskForm):
    name = StringField('Race Name', validators=[DataRequired(), Length(min=2, max=100)])
    date = DateTimeField('Race Date & Time', validators=[DataRequired()], default=datetime.now)
    location = StringField('Location', validators=[Optional(), Length(max=100)])
    distance = StringField('Distance', validators=[Optional(), Length(max=50)])
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
    purse = FloatField('Purse Amount', validators=[Optional(), NumberRange(min=0)])
    description = TextAreaField('Description', validators=[Optional(), Length(max=500)])
    submit = SubmitField('Create Race')

class HorseForm(FlaskForm):
    name = StringField('Horse Name', validators=[DataRequired(), Length(min=2, max=100)])
    age = IntegerField('Age', validators=[DataRequired(), NumberRange(min=2, max=20)])
    breed = StringField('Breed', validators=[Optional(), Length(max=50)])
    color = StringField('Color', validators=[Optional(), Length(max=30)])
    jockey = StringField('Jockey', validators=[Optional(), Length(max=100)])
    trainer = StringField('Trainer', validators=[Optional(), Length(max=100)])
    owner = StringField('Owner', validators=[Optional(), Length(max=100)])
    weight = FloatField('Weight (lbs)', validators=[Optional(), NumberRange(min=800, max=1400)])
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
    provider = StringField('Provider Name', validators=[DataRequired(), Length(min=2, max=50)])
    api_key = StringField('API Key', validators=[DataRequired(), Length(min=5, max=200)])
    api_secret = PasswordField('API Secret', validators=[Optional(), Length(max=200)])
    base_url = StringField('Base URL', validators=[Optional(), URL(), Length(max=200)])
    description = TextAreaField('Description', validators=[Optional(), Length(max=500)])
    is_active = BooleanField('Active', default=True)
    submit = SubmitField('Save Credentials')
    
    def validate_provider(self, field):
        # Check for valid provider name format
        if not field.data.replace('_', '').replace('-', '').isalnum():
            raise ValidationError('Provider name can only contain letters, numbers, hyphens, and underscores.')


class APICredentialsTestForm(FlaskForm):
    """Form for testing API credentials"""
    credential_id = HiddenField('Credential ID', validators=[DataRequired()])
    test_endpoint = StringField('Test Endpoint', validators=[Optional(), Length(max=200)])
    submit = SubmitField('Test Connection')