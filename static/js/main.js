/**
 * Main JavaScript file for HireFast application
 */

// Utility functions
const utils = {
    showNotification: function(message, type = 'info') {
        // Simple notification system
        const notification = document.createElement('div');
        notification.className = `fixed top-4 right-4 p-4 rounded-lg text-white z-50 ${
            type === 'error' ? 'bg-red-600' : 
            type === 'success' ? 'bg-green-600' : 
            'bg-blue-600'
        }`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    },
    
    formatDate: function(date) {
        return new Date(date).toLocaleDateString();
    },
    
    formatTime: function(date) {
        return new Date(date).toLocaleTimeString();
    }
};

// API helper functions
const api = {
    post: async function(url, data) {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    },
    
    get: async function(url) {
        const response = await fetch(url);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    }
};

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    console.log('HireFast application loaded');
    
    // Add any global initialization code here
    initializeApp();
});

function initializeApp() {
    // Check authentication status
    checkAuthStatus();
    
    // Set up event listeners
    setupEventListeners();
}

function setupEventListeners() {
    // Add any global event listeners here
    document.addEventListener('click', function(e) {
        // Close floating menu when clicking outside
        if (e.target.closest('.floating-action-container') === null) {
            const menu = document.getElementById('floatingMenu');
            if (menu && menu.classList.contains('show')) {
                menu.classList.remove('show');
            }
        }
    });
}

async function checkAuthStatus() {
    try {
        const data = await api.get('/check-auth');
        updateAuthStatus(data);
    } catch (error) {
        console.error('Failed to check auth status:', error);
        updateAuthStatus({ authenticated: false, error: error.message });
    }
}

function updateAuthStatus(authData) {
    const statusEl = document.getElementById('authStatus');
    if (!statusEl) return;
    
    if (authData.authenticated) {
        statusEl.className = 'inline-flex items-center px-4 py-2 rounded-full text-sm bg-green-600 text-white';
        statusEl.innerHTML = '<i class="fas fa-check-circle mr-2"></i><span>Authenticated</span>';
    } else {
        statusEl.className = 'inline-flex items-center px-4 py-2 rounded-full text-sm bg-red-600 text-white';
        statusEl.innerHTML = '<i class="fas fa-times-circle mr-2"></i><span>Not Authenticated</span>';
    }
}

// Export for use in other scripts
window.HireFast = {
    utils,
    api,
    checkAuthStatus,
    updateAuthStatus
};
