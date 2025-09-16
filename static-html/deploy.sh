#!/bin/bash

# Horse Racing Prediction - Static HTML Deployment Script

echo "🏇 Horse Racing Prediction - Static HTML Deployment"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "index.html" ]; then
    echo "❌ Error: index.html not found. Please run this script from the static-html directory."
    exit 1
fi

echo "✅ Found index.html - we're in the right directory"

# Function to start local development server
start_local_server() {
    echo "🚀 Starting local development server..."
    
    # Check for available tools and start server
    if command -v python3 &> /dev/null; then
        echo "📡 Using Python 3 HTTP server on port 8000"
        echo "🌐 Open http://localhost:8000 in your browser"
        python3 -m http.server 8000
    elif command -v python &> /dev/null; then
        echo "📡 Using Python 2 HTTP server on port 8000"
        echo "🌐 Open http://localhost:8000 in your browser"
        python -m SimpleHTTPServer 8000
    elif command -v php &> /dev/null; then
        echo "📡 Using PHP built-in server on port 8000"
        echo "🌐 Open http://localhost:8000 in your browser"
        php -S localhost:8000
    elif command -v npx &> /dev/null; then
        echo "📡 Using Node.js serve on port 3000"
        echo "🌐 Open http://localhost:3000 in your browser"
        npx serve . -p 3000
    else
        echo "❌ No suitable web server found. Please install Python, PHP, or Node.js"
        echo "💡 Or simply open index.html directly in your browser"
        exit 1
    fi
}

# Function to validate files
validate_files() {
    echo "🔍 Validating files..."
    
    required_files=(
        "index.html"
        "races.html"
        "add_race.html"
        "add_horse.html"
        "history.html"
        "stats.html"
        "api_import.html"
        "training.html"
        "css/style.css"
        "js/main.js"
        "js/data-manager.js"
        "js/app.js"
    )
    
    missing_files=()
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -eq 0 ]; then
        echo "✅ All required files are present"
        return 0
    else
        echo "❌ Missing files:"
        for file in "${missing_files[@]}"; do
            echo "   - $file"
        done
        return 1
    fi
}

# Function to create deployment package
create_package() {
    echo "📦 Creating deployment package..."
    
    # Create timestamp for unique filename
    timestamp=$(date +"%Y%m%d_%H%M%S")
    package_name="horse-racing-static-${timestamp}.zip"
    
    # Create zip file excluding unnecessary files
    zip -r "$package_name" . \
        -x "*.sh" \
        -x "*.git*" \
        -x "*.DS_Store" \
        -x "node_modules/*" \
        -x "*.log"
    
    echo "✅ Created deployment package: $package_name"
    echo "📤 Upload this file to your static hosting service"
}

# Function to deploy to GitHub Pages
deploy_github_pages() {
    echo "🚀 Deploying to GitHub Pages..."
    
    # Check if git is available
    if ! command -v git &> /dev/null; then
        echo "❌ Git is not installed. Please install Git first."
        return 1
    fi
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        echo "❌ Not a git repository. Please initialize git first:"
        echo "   git init"
        echo "   git remote add origin <your-github-repo-url>"
        return 1
    fi
    
    # Check for uncommitted changes
    if ! git diff-index --quiet HEAD --; then
        echo "📝 Found uncommitted changes. Adding and committing..."
        git add .
        git commit -m "Update static files for GitHub Pages deployment - $(date '+%Y-%m-%d %H:%M:%S')"
    else
        echo "✅ No uncommitted changes found"
    fi
    
    # Push to main branch
    echo "📤 Pushing to GitHub..."
    if git push origin main; then
        echo "✅ Successfully pushed to GitHub!"
        echo ""
        echo "🌐 Your site should be available at:"
        echo "   https://<username>.github.io/<repository-name>/"
        echo ""
        echo "📋 To enable GitHub Pages (if not already enabled):"
        echo "   1. Go to your repository on GitHub"
        echo "   2. Click on 'Settings' tab"
        echo "   3. Scroll down to 'Pages' section"
        echo "   4. Under 'Source', select 'Deploy from a branch'"
        echo "   5. Select 'main' branch and '/ (root)' folder"
        echo "   6. Click 'Save'"
        echo ""
    else
        echo "❌ Failed to push to GitHub. Please check your remote configuration."
        return 1
    fi
}

# Function to show deployment instructions
show_deployment_info() {
    echo ""
    echo "🚀 Deployment Options:"
    echo "====================="
    echo ""
    echo "1. 📁 GitHub Pages:"
    echo "   - Create a new repository on GitHub"
    echo "   - Upload all files to the repository"
    echo "   - Enable GitHub Pages in repository settings"
    echo ""
    echo "2. 🌐 Netlify:"
    echo "   - Go to https://netlify.com"
    echo "   - Drag and drop this folder to deploy"
    echo "   - Or connect your GitHub repository"
    echo ""
    echo "3. ⚡ Vercel:"
    echo "   - Go to https://vercel.com"
    echo "   - Import your GitHub repository"
    echo "   - Or use 'vercel' CLI command"
    echo ""
    echo "4. ☁️ AWS S3:"
    echo "   - Create an S3 bucket"
    echo "   - Enable static website hosting"
    echo "   - Upload all files to the bucket"
    echo ""
    echo "5. 🔥 Firebase Hosting:"
    echo "   - Install Firebase CLI: npm install -g firebase-tools"
    echo "   - Run: firebase init hosting"
    echo "   - Run: firebase deploy"
    echo ""
}

# Main menu
case "${1:-menu}" in
    "serve"|"start"|"dev")
        validate_files && start_local_server
        ;;
    "validate"|"check")
        validate_files
        ;;
    "package"|"zip")
        validate_files && create_package
        ;;
    "github"|"pages")
        validate_files && deploy_github_pages
        ;;
    "info"|"deploy")
        show_deployment_info
        ;;
    "menu"|*)
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  serve     Start local development server"
        echo "  validate  Check if all required files are present"
        echo "  package   Create deployment package (zip file)"
        echo "  github    Deploy to GitHub Pages"
        echo "  info      Show deployment information"
        echo ""
        echo "Examples:"
        echo "  $0 serve      # Start local server"
        echo "  $0 package    # Create deployment zip"
        echo "  $0 github     # Deploy to GitHub Pages"
        echo "  $0 info       # Show deployment options"
        ;;
esac