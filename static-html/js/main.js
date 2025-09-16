// Main Application JavaScript
class HorseRacingApp {
    constructor() {
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeTooltips();
        this.setupNavigation();
        this.loadInitialData();
    }

    setupEventListeners() {
        // Global event listeners
        document.addEventListener('DOMContentLoaded', () => {
            this.hideLoadingSpinners();
            this.setupFormValidation();
        });

        // Auto-hide alerts after 5 seconds
        document.querySelectorAll('.alert').forEach(alert => {
            if (!alert.classList.contains('alert-permanent')) {
                setTimeout(() => {
                    alert.style.opacity = '0';
                    setTimeout(() => alert.remove(), 300);
                }, 5000);
            }
        });
    }

    initializeTooltips() {
        // Initialize Bootstrap tooltips if available
        if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
            const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            tooltipTriggerList.map(function (tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        }
    }

    setupNavigation() {
        // Highlight current page in navigation
        const currentPage = window.location.pathname.split('/').pop() || 'index.html';
        const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
        
        navLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href === currentPage || (currentPage === '' && href === 'index.html')) {
                link.classList.add('active');
            }
        });

        // Smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                const href = this.getAttribute('href');
                // Check if href is not just '#' and contains a valid selector
                if (href && href.length > 1) {
                    const target = document.querySelector(href);
                    if (target) {
                        target.scrollIntoView({
                            behavior: 'smooth',
                            block: 'start'
                        });
                    }
                }
            });
        });
    }

    setupFormValidation() {
        // Bootstrap form validation
        const forms = document.querySelectorAll('.needs-validation');
        Array.from(forms).forEach(form => {
            form.addEventListener('submit', event => {
                if (!form.checkValidity()) {
                    event.preventDefault();
                    event.stopPropagation();
                }
                form.classList.add('was-validated');
            }, false);
        });
    }

    hideLoadingSpinners() {
        document.querySelectorAll('.loading-spinner').forEach(spinner => {
            spinner.style.display = 'none';
        });
    }

    loadInitialData() {
        // Load any initial data needed for the application
        this.checkDataIntegrity();
    }

    checkDataIntegrity() {
        // Ensure data structure is valid
        try {
            const races = DataManager.getRaces();
            const horses = DataManager.getHorses();
            
            // Validate data structure
            if (!Array.isArray(races)) {
                localStorage.removeItem('races');
            }
            if (!Array.isArray(horses)) {
                localStorage.removeItem('horses');
            }
        } catch (error) {
            console.warn('Data integrity check failed, clearing corrupted data:', error);
            DataManager.clearAllData();
        }
    }

    // Utility methods
    showAlert(message, type = 'info', container = null) {
        const alertHtml = `
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        const targetContainer = container || document.querySelector('.alert-container') || document.querySelector('.container').firstElementChild;
        if (targetContainer) {
            targetContainer.insertAdjacentHTML('afterbegin', alertHtml);
        }
    }

    showLoading(element) {
        if (element) {
            element.innerHTML = '<div class="text-center"><div class="spinner-border" role="status"><span class="visually-hidden">Loading...</span></div></div>';
        }
    }

    hideLoading(element, originalContent = '') {
        if (element) {
            element.innerHTML = originalContent;
        }
    }

    formatCurrency(amount) {
        return DataManager.formatCurrency(amount);
    }

    formatDate(dateString) {
        return DataManager.formatDate(dateString);
    }

    formatTime(dateString) {
        return DataManager.formatTime(dateString);
    }

    // API simulation methods
    async simulateAPICall(duration = 1000) {
        return new Promise(resolve => {
            setTimeout(resolve, duration);
        });
    }

    // Data export/import utilities
    exportData() {
        const data = DataManager.exportData();
        const blob = new Blob([data], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `horse-racing-data-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.showAlert('Data exported successfully!', 'success');
    }

    importData(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const success = DataManager.importData(e.target.result);
                    if (success) {
                        this.showAlert('Data imported successfully!', 'success');
                        // Refresh the page to show new data
                        setTimeout(() => window.location.reload(), 1000);
                        resolve(true);
                    } else {
                        this.showAlert('Failed to import data. Please check the file format.', 'danger');
                        reject(new Error('Import failed'));
                    }
                } catch (error) {
                    this.showAlert('Invalid file format. Please select a valid JSON file.', 'danger');
                    reject(error);
                }
            };
            reader.readAsText(file);
        });
    }

    // Chart utilities (for stats page)
    createChart(canvasId, type, data, options = {}) {
        const canvas = document.getElementById(canvasId);
        if (!canvas || typeof Chart === 'undefined') {
            console.warn('Chart.js not available or canvas not found');
            return null;
        }

        const ctx = canvas.getContext('2d');
        return new Chart(ctx, {
            type: type,
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                ...options
            }
        });
    }

    // Pagination utility
    paginate(items, page = 1, itemsPerPage = 10) {
        const startIndex = (page - 1) * itemsPerPage;
        const endIndex = startIndex + itemsPerPage;
        
        return {
            items: items.slice(startIndex, endIndex),
            totalPages: Math.ceil(items.length / itemsPerPage),
            currentPage: page,
            totalItems: items.length,
            hasNext: endIndex < items.length,
            hasPrev: page > 1
        };
    }

    // Search and filter utilities
    searchItems(items, searchTerm, searchFields = []) {
        if (!searchTerm) return items;
        
        const term = searchTerm.toLowerCase();
        return items.filter(item => {
            return searchFields.some(field => {
                const value = this.getNestedProperty(item, field);
                return value && value.toString().toLowerCase().includes(term);
            });
        });
    }

    getNestedProperty(obj, path) {
        return path.split('.').reduce((current, key) => current && current[key], obj);
    }

    // Date utilities
    isUpcoming(dateString) {
        return new Date(dateString) > new Date();
    }

    isPast(dateString) {
        return new Date(dateString) < new Date();
    }

    isToday(dateString) {
        const date = new Date(dateString);
        const today = new Date();
        return date.toDateString() === today.toDateString();
    }

    // Local storage utilities
    saveToLocalStorage(key, data) {
        try {
            localStorage.setItem(key, JSON.stringify(data));
            return true;
        } catch (error) {
            console.error('Failed to save to localStorage:', error);
            return false;
        }
    }

    loadFromLocalStorage(key, defaultValue = null) {
        try {
            const data = localStorage.getItem(key);
            return data ? JSON.parse(data) : defaultValue;
        } catch (error) {
            console.error('Failed to load from localStorage:', error);
            return defaultValue;
        }
    }
}

// Initialize the application
const app = new HorseRacingApp();

// Make app available globally
window.HorseRacingApp = app;