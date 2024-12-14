import { Component, OnInit } from '@angular/core';
import { ImageServiceService } from '../../shared/services/image-service.service';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-image-view',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './image-view.component.html',
  styleUrl: './image-view.component.css'
})
export class ImageViewComponent implements OnInit {
  
  constructor(private imageService: ImageServiceService) {}

  imageName: string = '';
  imageUrl: string | null = null; // Store the image URL

  ngOnInit(): void {
    
  }

  getImageByName(name: string): void {
    // Ensure the name ends with .jpg (you can change the extension if needed)
    if (!name.endsWith('.jpg')) {
      name += '.jpg';
    }

    this.imageService.getImagesByName(name).subscribe({
      next: (imageInfo) => {
        if (imageInfo && imageInfo.length > 0) {
          const imagePath = imageInfo[0].path; // Get the path of the image from the response
          this.imageUrl = `http://localhost:3000${imagePath}`; // Combine it with your server URL
        } else {
          console.log('No images found');
        }
      },
      error: (err) => {
        console.error('Error fetching images by name:', err);
      }
    });
  }
  
}
