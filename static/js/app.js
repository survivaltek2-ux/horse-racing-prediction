// Horse Racing Prediction App JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Auto-hide alerts after 5 seconds
    setTimeout(function() {
        var alerts = document.querySelectorAll('.alert');
        alerts.forEach(function(alert) {
            var bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);

    // Form validation feedback
    var forms = document.querySelectorAll('.needs-validation');
    Array.prototype.slice.call(forms).forEach(function(form) {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
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
                        behavior: 'smooth'
                    });
                }
            }
        });
    });
});

// Utility function to format currency
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(amount);
}

// Utility function to format dates
function formatDate(dateString) {
    const options = { 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    };
    return new Date(dateString).toLocaleDateString('en-US', options);
}

// Enhanced DateTime Picker functionality
function initializeDateTimePicker() {
    const datetimeInputs = document.querySelectorAll('.datetime-input');
    
    datetimeInputs.forEach(input => {
        // Set minimum date to current date/time
        const now = new Date();
        const minDateTime = now.toISOString().slice(0, 16);
        input.setAttribute('min', minDateTime);
        
        // Add visual feedback on interaction
        input.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
            const helper = this.nextElementSibling;
            if (helper && helper.classList.contains('datetime-helper')) {
                helper.style.opacity = '1';
            }
        });
        
        input.addEventListener('blur', function() {
            this.parentElement.classList.remove('focused');
            const helper = this.nextElementSibling;
            if (helper && helper.classList.contains('datetime-helper')) {
                helper.style.opacity = '0.8';
            }
        });
        
        // Validate selected date/time
        input.addEventListener('change', function() {
            const selectedDate = new Date(this.value);
            const currentDate = new Date();
            
            // Clear previous validation states
            this.classList.remove('is-invalid', 'is-valid');
            
            if (selectedDate < currentDate) {
                this.classList.add('is-invalid');
                showDateTimeError(this, 'Please select a future date and time');
            } else {
                this.classList.add('is-valid');
                clearDateTimeError(this);
                showDateTimePreview(this, selectedDate);
            }
        });
        
        // Add keyboard navigation hints
        input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.showPicker();
            }
        });
    });
}

// Show datetime error message
function showDateTimeError(input, message) {
    clearDateTimeError(input);
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'datetime-error text-danger mt-1';
    errorDiv.innerHTML = `<small><i class="fas fa-exclamation-triangle me-1"></i>${message}</small>`;
    
    input.parentElement.appendChild(errorDiv);
}

// Clear datetime error message
function clearDateTimeError(input) {
    const existingError = input.parentElement.querySelector('.datetime-error');
    if (existingError) {
        existingError.remove();
    }
}

// Show datetime preview
function showDateTimePreview(input, selectedDate) {
    clearDateTimePreview(input);
    
    const previewDiv = document.createElement('div');
    previewDiv.className = 'datetime-preview text-success mt-1';
    previewDiv.innerHTML = `
        <small>
            <i class="fas fa-check-circle me-1"></i>
            Race scheduled for: ${formatDate(selectedDate.toISOString())}
        </small>
    `;
    
    input.parentElement.appendChild(previewDiv);
}

// Clear datetime preview
function clearDateTimePreview(input) {
    const existingPreview = input.parentElement.querySelector('.datetime-preview');
    if (existingPreview) {
        existingPreview.remove();
    }
}

// Initialize datetime picker when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeDateTimePicker();
});