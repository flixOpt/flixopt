// Re-initialize Plotly charts on MkDocs Material instant navigation
document.addEventListener('DOMContentLoaded', function() {
    initPlotlyCharts();
});

// Hook into Material's instant navigation
if (typeof document$ !== 'undefined') {
    document$.subscribe(function() {
        initPlotlyCharts();
    });
}

function initPlotlyCharts() {
    const charts = document.querySelectorAll('div.mkdocs-plotly');
    charts.forEach(function(chart) {
        // Skip if already initialized (has children)
        if (chart.children.length > 0) return;

        try {
            const plotData = JSON.parse(chart.textContent);
            chart.textContent = '';
            const data = plotData.data || [];
            const layout = plotData.layout || {};
            const config = plotData.config || {responsive: true};
            Plotly.newPlot(chart, data, layout, config);
        } catch (e) {
            console.error('Failed to initialize Plotly chart:', e);
        }
    });
}
