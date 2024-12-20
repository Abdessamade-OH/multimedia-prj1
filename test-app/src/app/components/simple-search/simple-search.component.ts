import { Component } from '@angular/core';
import { ImageServiceService } from '../../shared/services/image-service.service';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { catchError, map, Observable, of } from 'rxjs';

@Component({
  selector: 'app-simple-search',
  standalone: true,
  imports: [FormsModule, CommonModule],
  templateUrl: './simple-search.component.html',
  styleUrls: ['./simple-search.component.css'],
})
export class SimpleSearchComponent {
  imageCategory!: string;
  numberK!: number;
  imageName: string = '';
  imageUrl: string | null = null;
  similarImages: any[] = [];
  isLoading: boolean = false; // Loading spinner state
  features: any = {}; // Store extracted features


  constructor(private imageService: ImageServiceService) {}

  getImageByName(name: string): void {
    console.log('Starting image search...');
    this.isLoading = true; // Start loading for image retrieval

    this.imageService.getImagesByName(name).subscribe({
      next: (imageInfo) => {
        console.log('Image search completed');
        if (imageInfo && imageInfo.length > 0) {
          const imagePath = imageInfo[0].path;
          const relativePath = imagePath.split('/src/upload_folder/')[1]; // Extract relative path
          this.imageUrl = `http://localhost:3000/uploaded_images/${relativePath}`;
          const imageNameFromPath = relativePath.split('/').pop();
          this.imageName = imageNameFromPath || 'Unknown';
          this.imageCategory = imageInfo[0].category;

          this.isLoading = false; // Stop loading after image retrieval
        } else {
          console.log('No image found');
          this.imageUrl = null;
          this.isLoading = false; // Stop loading on no image found
        }
      },
      error: (err) => {
        console.error('Error fetching image by name:', err);
        this.isLoading = false; // Stop loading on error
      },
    });
  }

  getImageByName2(name: string): Observable<any> {
    console.log('Starting image search...');
    this.isLoading = true; // Start loading for image retrieval

    return this.imageService.getImagesByName(name).pipe(
      map((imageInfo) => {
        console.log('Image search completed');
        if (imageInfo && imageInfo.length > 0) {
          const imagePath = imageInfo[0].path;
          const relativePath = imagePath.split('/src/upload_folder/')[1]; // Extract relative path
          const imageUrl = `http://localhost:3000/uploaded_images/${relativePath}`;
          const imageNameFromPath = relativePath.split('/').pop();
          const imageCategory = imageInfo[0].category;

          return { imageUrl, imageName: imageNameFromPath || 'Unknown', imageCategory };
        } else {
          console.log('No image found');
          return null; // Return null if no image found
        }
      }),
      catchError((err) => {
        console.error('Error fetching image by name:', err);
        return of(null); // Return null if there's an error
      })
    );
}
extractFeatures(): void {
  console.log('Starting feature extraction...');
  this.isLoading = true; // Start loading for feature extraction

  this.imageService.extractFeatures(this.imageName, this.imageCategory, this.numberK || 10).subscribe({
    next: (response) => {
      console.log(response);
      console.log('Feature extraction completed');
      this.similarImages = response.similar_images || []; // Extract similar images
      
      // Fetch the image for each similar image using getImageByName2
      let requestsCompleted = 0;
      const totalRequests = this.similarImages.length;

      this.similarImages.forEach((image: any) => {
        console.log(image);
        const imageName = this.extractImageName(image.image_path);
        this.getImageByName2(imageName).subscribe((imageData) => {
          if (imageData) {
            // Now you can use the imageData (imageUrl, imageName, imageCategory)
            console.log(imageData);
            // For example, you can update the image URL for each similar image
            image.url = imageData.imageUrl;
            image.name = imageData.imageName;
            this.features = response.features || {}; // Store extracted features
            console.log(this.features)
            image.category = imageData.imageCategory;
          }

          // Check if all requests have completed
          requestsCompleted++;
          if (requestsCompleted === totalRequests) {
            this.isLoading = false; // Stop loading after all requests are completed
          }
        });
      });

      // If no similar images were found, stop loading immediately
      if (totalRequests === 0) {
        this.isLoading = false;
      }
    },
    error: (error) => {
      console.error('Error extracting features:', error);
      this.isLoading = false; // Stop loading on error
    },
  });
}


  extractImageName(imagePath: string): string {
    const pathParts = imagePath.split('\\'); // Split the path by backslashes
    return pathParts[pathParts.length - 1]; // Return the last part (the name)
  }

  toggleSelection(image: any): void {
    image.isSelected = !image.isSelected;
  }

  performRelevanceSearch(): void {
    const irrelevantImages = this.similarImages.filter((img) => img.isIrrelevant);
    console.log('Irrelevant Images:', irrelevantImages);
    // Call your backend API here and pass the irrelevantImages
  }
  
}
