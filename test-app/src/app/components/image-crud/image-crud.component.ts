import { Component } from '@angular/core';
import { ImageServiceService } from '../../shared/services/image-service.service';
import { ReactiveFormsModule,FormsModule } from '@angular/forms';

@Component({
  selector: 'app-image-crud',
  standalone: true,
  imports: [ReactiveFormsModule,FormsModule],
  templateUrl: './image-crud.component.html',
  styleUrl: './image-crud.component.css'
})
export class ImageCrudComponent {
  imageCategory: string = ''; // Store selected category
  imageFiles: File[] = []; // Store selected image files
  deleteImageName: string = ''; // Store the image name for deletion
  errorMessage: string = ''; // Error message for duplicate names or other issues

  constructor(private imageService: ImageServiceService) {}

  // Handle file input changes
  onFileChange(event: any): void {
    const files = event.target.files;
    if (files.length > 0) {
      this.imageFiles = Array.from(files); // Convert FileList to an array
    } else {
      this.imageFiles = [];
    }
  }

  onUploadImage(): void {
    if (this.imageCategory && this.imageFiles.length > 0) {
      const formData = new FormData();
      this.imageFiles.forEach(file => formData.append('images', file)); // Append all files
      formData.append('category', this.imageCategory); // Append category
  
      this.imageService.uploadImage(formData).subscribe(
        (response) => {
          console.log('Images uploaded successfully:', response);
          alert('Upload successful!');
          this.resetForm();
        },
        (error) => {
          console.error('Error uploading images:', error);
          const errorMessage = error.error?.error || 'An unexpected error occurred. Please try again.';
          alert(errorMessage);
        }
      );
    } else {
      alert('Please select a category and at least one image.');
    }
  }
  
  // Delete image by name
  onDeleteImage(): void {
    if (this.deleteImageName) {
      this.imageService.deleteImage(this.deleteImageName).subscribe(
        (response) => {
          console.log('Image deleted successfully:', response);
          alert('Image deleted successfully!');
        },
        (error) => {
          console.error('Error deleting image:', error);
          alert('Error deleting image. Please try again.');
        }
      );
    } else {
      alert('Please provide the name of the image to delete.');
    }
  }

  // Reset form after successful upload
  private resetForm(): void {
    this.imageFiles = [];
    const fileInput = document.getElementById('image-file') as HTMLInputElement;
    if (fileInput) {
      fileInput.value = ''; // Reset the file input
    }
    this.imageCategory = ''; // Reset category selection
  }
  
}
