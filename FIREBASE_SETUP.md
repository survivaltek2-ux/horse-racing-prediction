# Firebase Setup Guide

This guide will help you set up Google Firebase as the database for the Horse Racing Prediction application.

## Prerequisites

1. Google Cloud Platform account
2. Firebase project created
3. Firebase Admin SDK service account key

## Step 1: Create Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Create a project" or "Add project"
3. Enter your project name (e.g., "horse-racing-prediction")
4. Enable Google Analytics (optional)
5. Click "Create project"

## Step 2: Enable Firestore Database

1. In your Firebase project console, go to "Firestore Database"
2. Click "Create database"
3. Choose "Start in test mode" (you can configure security rules later)
4. Select a location for your database
5. Click "Done"

## Step 3: Generate Service Account Key

1. Go to Project Settings (gear icon) → "Service accounts"
2. Click "Generate new private key"
3. Download the JSON file
4. Rename it to `firebase-service-account.json`
5. Place it in the `config/` directory of your application

## Step 4: Configure Environment Variables

1. Copy `.env.example` to `.env`
2. Update the following variables:

```bash
# Firebase Configuration
FIREBASE_SERVICE_ACCOUNT_PATH=config/firebase-service-account.json
ENCRYPTION_KEY=your-32-character-encryption-key-here
```

**Important**: Generate a secure 32-character encryption key for encrypting API credentials.

## Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 6: Run Migration (Optional)

If you have existing data in SQLite or JSON files:

```bash
python migrate_to_firebase.py
```

This script will:
- Migrate existing JSON data to Firebase
- Migrate SQLite database data to Firebase
- Create sample data if no existing data is found

## Step 7: Start the Application

```bash
python app.py
```

The application will automatically:
- Initialize Firebase connection
- Create admin user if it doesn't exist
- Set up Firestore collections

## Firestore Collections Structure

The application uses the following Firestore collections:

### api_credentials
- Stores encrypted API keys and credentials
- Fields: provider, api_key (encrypted), api_secret (encrypted), base_url, description, is_active

### races
- Stores race information
- Fields: name, date, track, distance, surface, conditions, horses, results

### horses
- Stores horse information
- Fields: name, age, weight, jockey, trainer, odds, form, stats

### predictions
- Stores prediction results
- Fields: race_id, horse_id, predicted_position, confidence, actual_position, created_at

### users
- Stores user accounts
- Fields: username, email, password_hash, is_admin, created_at

## Security Considerations

1. **Service Account Key**: Keep your `firebase-service-account.json` file secure and never commit it to version control
2. **Encryption Key**: Use a strong, randomly generated encryption key
3. **Firestore Rules**: Configure appropriate security rules for production use
4. **Environment Variables**: Never commit `.env` file with real credentials

## Firestore Security Rules (Production)

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Users can only access their own data
    match /users/{userId} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
    
    // Admin-only collections
    match /api_credentials/{document} {
      allow read, write: if request.auth != null && 
        get(/databases/$(database)/documents/users/$(request.auth.uid)).data.is_admin == true;
    }
    
    // Public read access for races and horses
    match /races/{document} {
      allow read: if true;
      allow write: if request.auth != null;
    }
    
    match /horses/{document} {
      allow read: if true;
      allow write: if request.auth != null;
    }
    
    // Users can read all predictions, but only create their own
    match /predictions/{document} {
      allow read: if true;
      allow create: if request.auth != null;
      allow update, delete: if request.auth != null && 
        resource.data.user_id == request.auth.uid;
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **Firebase initialization failed**: Check that your service account file path is correct
2. **Permission denied**: Ensure Firestore is enabled and security rules allow access
3. **Encryption errors**: Verify your encryption key is exactly 32 characters
4. **Import errors**: Make sure all required packages are installed

### Logs

Check the application logs for detailed error messages:
- Console output when starting the application
- Check `logs/app.log` if logging is configured

## Migration from SQLite

The migration script (`migrate_to_firebase.py`) handles:
- Converting SQLite data to Firestore documents
- Preserving relationships between entities
- Encrypting sensitive data (API credentials)
- Creating proper document IDs and timestamps

## Benefits of Firebase

1. **Scalability**: Automatically scales with your application
2. **Real-time**: Real-time updates and synchronization
3. **Security**: Built-in authentication and security rules
4. **Backup**: Automatic backups and point-in-time recovery
5. **Global**: Multi-region deployment capabilities
6. **Integration**: Easy integration with other Google Cloud services