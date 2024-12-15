import { Component, OnInit } from '@angular/core';
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
export class SimpleSearchComponent{

  constructor(private imageService: ImageServiceService) {}

  imageName: string = '';
  imageUrl: string | null = null;


  getImageByName(name: string): void {
    // Ensure the name ends with .jpg
    /*if (!name.endsWith('.jpg')) {
      name += '.jpg';
    }*/

    // Fetch image by name
    this.imageService.getImagesByName(name).subscribe({
      next: (imageInfo) => {
        if (imageInfo && imageInfo.length > 0) {
          const imagePath = imageInfo[0].path; // Full path stored in the database
          const relativePath = imagePath.split('/src/upload_folder/')[1]; // Extract relative path after upload_folder
          this.imageUrl = `http://localhost:3000/uploaded_images/${relativePath}`;
          console.log(imageInfo);
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
}