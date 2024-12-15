import { Component } from '@angular/core';
import { ImageServiceService } from '../../shared/services/image-service.service';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-simple-search',
  standalone: true,
  imports: [FormsModule, CommonModule],
  templateUrl: './simple-search.component.html',
  styleUrl: './simple-search.component.css'
})
export class SimpleSearchComponent {
  imageCategory!: string;
  numberK!: number;
  imageName: string = '';
  imageUrl: string | null = null;

  constructor(private imageService: ImageServiceService) {}

  // Fetch the image by name (same as before)
  // In your SimpleSearchComponent
getImageByName(name: string): void {
  this.imageService.getImagesByName(name).subscribe({
    next: (imageInfo) => {
      if (imageInfo && imageInfo.length > 0) {
        const imagePath = imageInfo[0].path;
        const relativePath = imagePath.split('/src/upload_folder/')[1]; // Extract relative path
        this.imageUrl = `http://localhost:3000/uploaded_images/${relativePath}`;
        
        // Extract the image name from the path (last part after the last slash)
        const imageNameFromPath = relativePath.split('/').pop();
        this.imageName = imageNameFromPath ? imageNameFromPath : 'Unknown';  // Store the name
        this.imageCategory = imageInfo[0].category;
      } else {
        this.imageUrl = null;
        console.log('No image found');
      }
    },
    error: (err) => {
      console.error('Error fetching image by name:', err);
      this.imageUrl = null;
    }
  });
}

extractFeatures(): void {
  this.imageService.extractFeatures(this.imageName, this.imageCategory, 10).subscribe({
    next: (response) => {
      console.log('Features extracted successfully', response);
      // Handle the response (show the results or do further processing)
    },
    error: (error) => {
      console.error('Error extracting features:', error);
      // Handle the error (show an error message, etc.)
    }
  });
}

}
