from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, FloatField, TextAreaField, SelectField, DateTimeField, SubmitField
from wtforms.validators import DataRequired, NumberRange, Length, Optional
from datetime import datetime

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