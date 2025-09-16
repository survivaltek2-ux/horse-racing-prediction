# Horse Racing Prediction - Static HTML Version

This is a static HTML version of the Horse Racing Prediction application, converted from the original Flask-based web application. It can be hosted on any static web server without requiring a backend.

**Last Deployment:** January 17, 2025 - Updated with async data loading and authentication considerations

## Features

- **Dashboard**: Overview of races, horses, and statistics
- **Race Management**: Add, view, and manage horse races
- **Horse Management**: Add and manage horse information
- **Predictions**: Generate predictions for races using a client-side algorithm
- **Model Training**: Train prediction models with custom parameters and monitoring
- **Statistics**: View comprehensive statistics and charts
- **API Import**: Simulate importing races from external APIs
- **History**: View past races and results
- **Data Persistence**: All data is stored locally in the browser's localStorage

## File Structure

```
static-html/
├── index.html          # Main dashboard page
├── races.html          # Race listing and management
├── add_race.html       # Add new race form
├── add_horse.html      # Add new horse form
├── history.html        # Race history and results
├── stats.html          # Statistics and charts
├── api_import.html     # API import simulation
├── training.html       # Model training interface
├── css/
│   └── style.css       # Custom styles
├── js/
│   ├── main.js         # Main application logic
│   ├── data-manager.js # Data management and storage
│   └── app.js          # Additional utilities
├── data/
│   ├── races.json      # Sample race data
│   └── horses.json     # Sample horse data
└── img/                # Images directory
```

## Technologies Used

- **HTML5**: Semantic markup
- **CSS3**: Styling with Bootstrap 5
- **JavaScript (ES6+)**: Client-side functionality
- **Bootstrap 5**: UI framework
- **Font Awesome**: Icons
- **Chart.js**: Statistics charts (optional)
- **LocalStorage**: Data persistence

## Setup and Deployment

### Local Testing
1. Navigate to the static-html directory:
   ```bash
   cd /path/to/static-html
   ```

2. **Option A: Using the deployment script**
   ```bash
   ./deploy.sh
   ```
   Choose option 1 to start a local server

3. **Option B: Manual server start**
   ```bash
   # Python 3
   python3 -m http.server 8000
   
   # Python 2
   python -m SimpleHTTPServer 8000
   
   # Node.js (if you have it installed)
   npx serve .
   
   # PHP (if available)
   php -S localhost:8000
   ```

4. Open your browser to `http://localhost:8000`

### Testing Checklist
After starting the local server, verify these features work:
- ✅ Navigation between all pages
- ✅ Add new races with form validation
- ✅ Add new horses with form validation
- ✅ View race history with filtering
- ✅ Generate predictions for races
- ✅ Model training interface and monitoring
- ✅ View statistics and charts
- ✅ API import simulation
- ✅ Data export/import functionality
- ✅ Responsive design on mobile devices

### Production Deployment

#### Static Hosting Services
The application can be deployed to any static hosting service:

**GitHub Pages:**
1. Push the static-html folder contents to a GitHub repository
2. Enable GitHub Pages in repository settings
3. Select the main branch as source

**Netlify:**
1. Drag and drop the static-html folder to Netlify
2. Or connect your GitHub repository
3. Build command: (none needed)
4. Publish directory: `/` (root)

**Vercel:**
1. Import your GitHub repository
2. Framework preset: Other
3. Build command: (none needed)
4. Output directory: `./`

**AWS S3 + CloudFront:**
1. Create an S3 bucket with static website hosting
2. Upload all files from static-html directory
3. Set up CloudFront distribution for HTTPS and CDN

#### Traditional Web Servers

**Apache:**
1. Copy files to your web directory (e.g., `/var/www/html/`)
2. Ensure `.htaccess` is allowed if you need URL rewriting
3. No special configuration needed

**Nginx:**
```nginx
server {
    listen 80;
    server_name your-domain.com;
    root /path/to/static-html;
    index index.html;
    
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

### Environment Configuration

#### API Integration
To connect to real racing APIs, modify the API configuration in:
- `js/data-manager.js` - Update the `fetchRacesFromAPI` method
- `api_import.html` - Configure API endpoints and authentication

#### Data Persistence
Currently uses localStorage. For production with multiple users:
1. Replace localStorage calls with API calls to your backend
2. Implement user authentication
3. Set up a database for persistent storage

### Performance Optimization

#### For Production
1. **Minify JavaScript and CSS:**
   ```bash
   # Using terser for JS
   npm install -g terser
   terser js/app.js -o js/app.min.js
   terser js/data-manager.js -o js/data-manager.min.js
   
   # Using cssnano for CSS
   npm install -g cssnano-cli
   cssnano css/style.css css/style.min.css
   ```

2. **Enable Gzip compression** on your web server

3. **Set up proper caching headers:**
   ```apache
   # Apache .htaccess
   <IfModule mod_expires.c>
       ExpiresActive on
       ExpiresByType text/css "access plus 1 year"
       ExpiresByType application/javascript "access plus 1 year"
       ExpiresByType image/png "access plus 1 year"
   </IfModule>
   ```

### Monitoring and Analytics
Add analytics tracking by including your tracking code in each HTML file:
```html
<!-- Google Analytics example -->
<script async src="https://www.googletagmanager.com/gtag/js?id=GA_TRACKING_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'GA_TRACKING_ID');
</script>
```

## Data Management

### Local Storage
- All data is stored in the browser's localStorage
- Data persists between sessions
- No server-side database required

### Data Structure
- **Races**: Stored as JSON array with race details, horses, and results
- **Horses**: Stored as JSON array with horse information
- **Predictions**: Stored within race objects

### Import/Export
- Export data as JSON file for backup
- Import data from JSON file to restore or migrate

## API Simulation

The application includes a simulated API import feature that:
- Generates realistic sample race data
- Simulates API delays and responses
- Allows importing races from "external" sources
- Demonstrates how real API integration would work

## Prediction Algorithm

The application includes a simple prediction algorithm that considers:
- Horse odds (lower odds = higher chance)
- Horse weight (lighter weight = advantage)
- Random factors for realism
- Confidence scoring (0.1 to 0.95)

## Browser Compatibility

- Modern browsers with ES6+ support
- Chrome 60+
- Firefox 55+
- Safari 12+
- Edge 79+

## Limitations

- Data is stored locally (not shared between devices/browsers)
- No real-time updates or collaboration
- Limited to client-side processing
- No user authentication or multi-user support

## Future Enhancements

- Progressive Web App (PWA) features
- Offline functionality
- Real API integrations
- Advanced prediction algorithms
- Data synchronization options

## License

This project is part of the Horse Racing Prediction application suite.