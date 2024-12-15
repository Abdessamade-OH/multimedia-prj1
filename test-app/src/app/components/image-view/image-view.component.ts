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

  imageName: string = '';
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
        // Prepend localhost URL to each image path
        this.categoryImages = images.map((image: { path: any; }) => ({
          ...image,
          path: `http://localhost:3000${image.path}` // Add the localhost to the image path
        }));
      },
      error: (err) => {
        console.error('Error fetching images by category:', err);
        this.categoryImages = []; // Reset the images if there's an error
      }
    });
  }
  

  getImageByName(name: string): void {
    // Reset category to null to hide category-based images
    this.selectedCategory = null;
    this.categoryImages = []; // Clear category images

    // Ensure the name ends with .jpg
    if (!name.endsWith('.jpg')) {
      name += '.jpg';
    }

    // Fetch image by name
    this.imageService.getImagesByName(name).subscribe({
      next: (imageInfo) => {
        if (imageInfo && imageInfo.length > 0) {
          const imagePath = imageInfo[0].path;
          this.imageUrl = `http://localhost:3000${imagePath}`;
          console.log(imageInfo)
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
