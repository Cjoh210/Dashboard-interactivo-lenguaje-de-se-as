// red-line-plugin.js
Chart.register({
  id: 'redLinePlugin',
  afterDraw(chart, args) {
    const ctx = chart.ctx;
    const xScale = chart.scales.x;

    // Calcular la posición en píxeles de la línea roja
    const time = chart.getDatasetMeta(0).data[chart.data.labels.length - 1].x;
    const pixel = xScale.getValueForPixel(time);

    // Dibujar la línea roja
    ctx.save();
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pixel, chart.chartArea.top);
    ctx.lineTo(pixel, chart.chartArea.bottom);
    ctx.stroke();
    ctx.restore();
  }
});