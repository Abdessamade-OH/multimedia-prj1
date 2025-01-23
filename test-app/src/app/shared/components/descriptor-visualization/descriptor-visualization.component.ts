import { Component, OnInit, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

@Component({
  selector: 'app-descriptor-visualization',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div style="width: 80%; margin: auto;">
      <canvas #fourierChart></canvas>
      <hr />
      <canvas #zernikeChart></canvas>
    </div>
  `
})
export class DescriptorVisualizationComponent implements OnInit {
  @Input() fourier: number[] = [];
  @Input() zernike: number[] = [];

  ngOnInit(): void {
    this.createFourierChart();
    this.createZernikeChart();
  }

  createFourierChart(): void {
    const ctx = document.querySelector('canvas') as HTMLCanvasElement;
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: this.fourier.map((_, index) => `Point ${index + 1}`),
        datasets: [{
          label: 'Fourier Descriptors',
          data: this.fourier,
          borderColor: 'blue',
          tension: 0.1,
          pointStyle: 'line'
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: false
          }
        }
      }
    });
  }

  createZernikeChart(): void {
    const ctx = document.querySelectorAll('canvas')[1] as HTMLCanvasElement;
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: this.zernike.map((_, index) => `Point ${index + 1}`),
        datasets: [{
          label: 'Zernike Descriptors',
          data: this.zernike,
          borderColor: 'green',
          tension: 0.1,
          pointStyle: 'line'
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: false
          }
        }
      }
    });
  }
}