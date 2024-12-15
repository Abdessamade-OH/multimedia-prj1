import { Component, OnInit } from '@angular/core';
import { ImageServiceService } from '../../shared/services/image-service.service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-image-view',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './image-view.component.html',
  styleUrls: ['./image-view.component.css']
})
export class ImageViewComponent implements OnInit {

  constructor(private imageService: ImageServiceService) {}

  imageUrl: string | null = null;
  selectedCategory: string | null = 'aGrass'; // Default category
  categoryImages: any[] = []; // To hold images fetched by category

  ngOnInit(): void {
    // Fetch images for the default category
    this.getImagesByCategory(this.selectedCategory!);
  }

  selectCategory(category: string): void {
    this.selectedCategory = category;
    this.getImagesByCategory(category);
  }

  getImagesByCategory(category: string): void {
    // Reset the imageUrl when fetching by category
    this.imageUrl = null;
  
    // Fetch images by category from the backend
    this.imageService.getImagesByCategory(category).subscribe({
      next: (images) => {
        // Prepend localhost URL to each image path with the relative path extraction
        this.categoryImages = images.map((image: { path: string }) => {
          const relativePath = image.path.split('/src/upload_folder/')[1]; // Extract relative path
          return {
            ...image,
            path: `http://localhost:3000/uploaded_images/${relativePath}` // Full URL to image
          };
        });
      },
      error: (err) => {
        console.error('Error fetching images by category:', err);
        this.categoryImages = []; // Reset the images if there's an error
      }
    });
  }
  

  deleteImage(id: string): void {
    this.imageService.deleteImageById(id).subscribe({
      next: () => {
        console.log('Image deleted successfully');
        
        // Refresh the category images
        if (this.selectedCategory) {
          this.getImagesByCategory(this.selectedCategory);
        }
      },
      error: (err) => {
        console.error('Error deleting image:', err);
      }
    });
  }
  
  
}
