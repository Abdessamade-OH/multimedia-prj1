import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { 
  BarChart, Bar, XAxis, YAxis, 
  CartesianGrid, Tooltip, Legend 
} from 'recharts';

@Component({
  selector: 'app-color-histogram',
  standalone: true,
  imports: [CommonModule],
  template: `
    <BarChart width={400} height={300} data={chartData}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="name" />
      <YAxis />
      <Tooltip />
      <Legend />
      <Bar dataKey="red" fill="#ff0000" fillOpacity={0.6} />
      <Bar dataKey="green" fill="#00ff00" fillOpacity={0.6} />
      <Bar dataKey="blue" fill="#0000ff" fillOpacity={0.6} />
    </BarChart>
  `
})
export class ColorHistogramComponent {
  @Input() set histogramData(data: any) {
    if (data) {
      this.chartData = Array(Math.max(
        data.red.length,
        data.green.length,
        data.blue.length
      )).fill(0).map((_, i) => ({
        name: `Bin ${i}`,
        red: data.red[i] || 0,
        green: data.green[i] || 0,
        blue: data.blue[i] || 0
      }));
    }
  }
  
  chartData: any[] = [];
}