// VIVESPACES-AI/search_tracker.js

/**
 * ViveSpaces Search Tracker
 * Captura eventos de búsqueda en tiempo real desde el frontend
 */

class ViveSpacesSearchTracker {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.userId = this.getCurrentUserId();
        this.searchStartTime = null;
        this.currentQuery = '';
        this.apiEndpoint = '/api/ai/track';
        
        // Inicializar tracking automático
        this.initializeTracking();
    }
    
    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
    
    getCurrentUserId() {
        // Obtener user_id desde Laravel (si está disponible)
        const userDataElement = document.getElementById('user-data');
        if (userDataElement) {
            try {
                const userData = JSON.parse(userDataElement.textContent || userDataElement.innerText);
                return userData?.id || null;
            } catch (e) {
                console.log('No se pudo obtener user_id');
                return null;
            }
        }
        return null;
    }
    
    getDeviceType() {
        const width = window.innerWidth;
        if (width <= 768) return 'mobile';
        if (width <= 1024) return 'tablet';
        return 'desktop';
    }
    
    initializeTracking() {
        // Tracking de búsquedas en formularios
        this.trackSearchForms();
        
        // Tracking de filtros
        this.trackFilters();
        
        // Tracking de clicks en resultados
        this.trackResultClicks();
        
        // Tracking de tiempo en página
        this.trackPageTime();
        
        console.log('🔍 ViveSpaces Search Tracker inicializado');
    }
    
    trackSearchForms() {
        // Buscar todos los formularios de búsqueda
        const searchForms = document.querySelectorAll('form[action*="properties"], .search-form, #search-form');
        const searchInputs = document.querySelectorAll('input[name="search"], input[placeholder*="buscar"], .search-input');
        
        // Tracking de envío de formularios
        searchForms.forEach(form => {
            form.addEventListener('submit', (e) => {
                const formData = new FormData(form);
                const searchQuery = formData.get('search') || formData.get('query') || '';
                
                if (searchQuery.trim()) {
                    this.trackSearchEvent({
                        query: searchQuery.trim(),
                        type: 'form_submit',
                        filters: this.extractFiltersFromForm(form)
                    });
                }
            });
        });
        
        // Tracking de búsquedas en tiempo real (mientras escribe)
        searchInputs.forEach(input => {
            let searchTimeout;
            
            input.addEventListener('input', (e) => {
                clearTimeout(searchTimeout);
                
                // Esperar 1 segundo después de que pare de escribir
                searchTimeout = setTimeout(() => {
                    const query = e.target.value.trim();
                    
                    if (query.length >= 3) { // Mínimo 3 caracteres
                        this.trackSearchEvent({
                            query: query,
                            type: 'live_search',
                            filters: this.extractFiltersFromPage()
                        });
                    }
                }, 1000);
            });
            
            // Tracking cuando comienza a escribir
            input.addEventListener('focus', () => {
                this.searchStartTime = Date.now();
            });
        });
    }
    
    trackFilters() {
        // Tracking de filtros (select, checkboxes, radio buttons)
        const filterElements = document.querySelectorAll('select[name*="type"], select[name*="city"], select[name*="price"], input[name*="bedrooms"], input[name*="bathrooms"]');
        
        filterElements.forEach(element => {
            element.addEventListener('change', (e) => {
                this.trackSearchEvent({
                    query: this.getCurrentSearchQuery(),
                    type: 'filter_change',
                    filters: this.extractFiltersFromPage(),
                    filter_changed: {
                        name: e.target.name,
                        value: e.target.value
                    }
                });
            });
        });
    }
    
    trackResultClicks() {
        // Tracking de clicks en resultados de propiedades
        document.addEventListener('click', (e) => {
            // Buscar si el click fue en una propiedad
            const propertyCard = e.target.closest('.property-card, .property-item, [data-property-id]');
            
            if (propertyCard) {
                const propertyId = propertyCard.getAttribute('data-property-id') || 
                                 propertyCard.querySelector('[data-property-id]')?.getAttribute('data-property-id');
                
                const clickPosition = this.getClickPosition(propertyCard);
                
                this.trackSearchEvent({
                    query: this.getCurrentSearchQuery(),
                    type: 'result_click',
                    property_id: propertyId,
                    click_position: clickPosition,
                    element_clicked: e.target.tagName.toLowerCase() + (e.target.className ? '.' + e.target.className.replace(/\s+/g, '.') : '')
                });
            }
        });
    }
    
    trackPageTime() {
        // Tracking de tiempo en página de resultados
        this.pageStartTime = Date.now();
        
        // Enviar tiempo cuando el usuario sale de la página
        window.addEventListener('beforeunload', () => {
            const timeSpent = Math.round((Date.now() - this.pageStartTime) / 1000);
            
            if (timeSpent > 5) { // Solo si estuvo más de 5 segundos
                this.trackSearchEvent({
                    query: this.getCurrentSearchQuery(),
                    type: 'page_time',
                    time_spent: timeSpent
                }, true); // Envío síncrono
            }
        });
        
        // Tracking de scroll depth
        let maxScrollDepth = 0;
        window.addEventListener('scroll', () => {
            const scrollDepth = (window.scrollY + window.innerHeight) / document.body.scrollHeight;
            maxScrollDepth = Math.max(maxScrollDepth, scrollDepth);
        });
        
        // Enviar scroll depth cada 30 segundos
        setInterval(() => {
            if (maxScrollDepth > 0.1) { // Solo si scrolleó algo
                this.trackSearchEvent({
                    query: this.getCurrentSearchQuery(),
                    type: 'scroll_depth',
                    scroll_depth: Math.round(maxScrollDepth * 100)
                });
            }
        }, 30000);
    }
    
    extractFiltersFromForm(form) {
        const filters = {};
        const formData = new FormData(form);
        
        for (let [key, value] of formData.entries()) {
            if (key !== 'search' && key !== 'query' && value) {
                filters[key] = value;
            }
        }
        
        return filters;
    }
    
    extractFiltersFromPage() {
        const filters = {};
        
        // Extraer filtros comunes
        const filterSelectors = {
            'type': 'select[name="type"], select[name="property_type"]',
            'city': 'select[name="city"], input[name="city"]',
            'min_price': 'input[name="min_price"], input[name="price_min"]',
            'max_price': 'input[name="max_price"], input[name="price_max"]',
            'bedrooms': 'select[name="bedrooms"], input[name="bedrooms"]',
            'bathrooms': 'select[name="bathrooms"], input[name="bathrooms"]'
        };
        
        Object.entries(filterSelectors).forEach(([key, selector]) => {
            const element = document.querySelector(selector);
            if (element && element.value) {
                filters[key] = element.value;
            }
        });
        
        return filters;
    }
    
    getCurrentSearchQuery() {
        // Obtener query actual de la página
        const searchInput = document.querySelector('input[name="search"], input[name="query"], .search-input');
        if (searchInput) {
            return searchInput.value.trim();
        }
        
        // Obtener de URL si no hay input
        const urlParams = new URLSearchParams(window.location.search);
        return urlParams.get('search') || urlParams.get('query') || '';
    }
    
    getClickPosition(propertyCard) {
        // Obtener posición del elemento clickeado en la lista
        const allCards = document.querySelectorAll('.property-card, .property-item, [data-property-id]');
        return Array.from(allCards).indexOf(propertyCard) + 1;
    }
    
    getResultsCount() {
        // Contar resultados en la página
        const resultCards = document.querySelectorAll('.property-card, .property-item, [data-property-id]');
        return resultCards.length;
    }
    
    trackSearchEvent(eventData, isSync = false) {
        const searchEvent = {
            user_id: this.userId,
            session_id: this.sessionId,
            search_query: eventData.query || '',
            search_type: eventData.type || 'general',
            filters: eventData.filters || {},
            results_count: eventData.results_count || this.getResultsCount(),
            click_position: eventData.click_position || null,
            time_spent: eventData.time_spent || null,
            device: this.getDeviceType(),
            page_url: window.location.href,
            referrer: document.referrer,
            scroll_depth: eventData.scroll_depth || null,
            property_id: eventData.property_id || null,
            element_clicked: eventData.element_clicked || null,
            filter_changed: eventData.filter_changed || null,
            timestamp: new Date().toISOString()
        };
        
        // Enviar al servidor
        this.sendToServer(searchEvent, isSync);
        
        // 🔥 NUEVO: Programar procesamiento de recomendaciones después de 2 minutos
        // Solo si el usuario está autenticado y la búsqueda tiene al menos 3 caracteres
        if (this.userId && eventData.query && eventData.query.length >= 3) {
            setTimeout(() => {
                this.processRecommendations(eventData.query, eventData.type || 'general');
            }, 120000); // 2 minutos = 120000 ms
        }
        
        // Log para debugging (remover en producción)
        console.log('🔍 Search Event:', eventData.type, eventData.query);
    }
    
    sendToServer(eventData, isSync = false) {
        const url = this.apiEndpoint;
        const data = JSON.stringify(eventData);
        
        if (isSync) {
            // Envío síncrono para eventos al salir de la página
            navigator.sendBeacon(url, data);
        } else {
            // Envío asíncrono normal
            fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Requested-With': 'XMLHttpRequest',
                    'X-CSRF-TOKEN': this.getCSRFToken()
                },
                body: data
            }).catch(error => {
                console.log('Error enviando evento de búsqueda:', error);
            });
        }
    }
    
    /**
     * 🔥 NUEVO: Procesar recomendaciones después de una búsqueda
     * Se llama 2 minutos después de trackear la búsqueda
     */
    processRecommendations(searchQuery, searchType = 'general') {
        // Solo procesar si hay usuario autenticado
        if (!this.userId) {
            console.log('⚠️ Usuario no autenticado, no se procesarán recomendaciones');
            return;
        }

        const url = '/search/process-recommendations';
        const data = JSON.stringify({
            search_query: searchQuery,
            search_type: searchType
        });

        console.log('🔄 Procesando recomendaciones para:', searchQuery);

        fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest',
                'X-CSRF-TOKEN': this.getCSRFToken()
            },
            body: data
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log('✅ Recomendaciones procesadas:', data.message);
                console.log('📊 Total de recomendaciones:', data.recommendations_count);
            } else {
                console.log('⚠️ No se pudieron procesar recomendaciones:', data.message);
            }
        })
        .catch(error => {
            console.log('❌ Error procesando recomendaciones:', error);
        });
    }
    
    getCSRFToken() {
        const token = document.querySelector('meta[name="csrf-token"]');
        return token ? token.getAttribute('content') : '';
    }
    
    // Métodos públicos para tracking manual
    trackCustomSearch(query, filters = {}) {
        this.trackSearchEvent({
            query: query,
            type: 'custom',
            filters: filters
        });
    }
    
    trackPropertyView(propertyId) {
        this.trackSearchEvent({
            query: this.getCurrentSearchQuery(),
            type: 'property_view',
            property_id: propertyId
        });
    }
    
    trackContactClick(propertyId) {
        this.trackSearchEvent({
            query: this.getCurrentSearchQuery(),
            type: 'contact_click',
            property_id: propertyId
        });
    }
}

// Inicializar tracker automáticamente cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', function() {
    // Solo inicializar en páginas de propiedades
    if (window.location.pathname.includes('/properties') || 
        document.querySelector('.search-form, #search-form, .property-card')) {
        
        window.vivespacesTracker = new ViveSpacesSearchTracker();
        
        // Hacer disponible globalmente para uso manual
        window.trackSearch = (query, filters) => {
            window.vivespacesTracker.trackCustomSearch(query, filters);
        };
        
        window.trackPropertyView = (propertyId) => {
            window.vivespacesTracker.trackPropertyView(propertyId);
        };
        
        window.trackContactClick = (propertyId) => {
            window.vivespacesTracker.trackContactClick(propertyId);
        };
    }
});

// Exportar para uso en módulos
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ViveSpacesSearchTracker;
}