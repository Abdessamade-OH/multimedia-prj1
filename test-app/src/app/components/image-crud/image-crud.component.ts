import { Component } from '@angular/core';
import { ImageServiceService } from '../../shared/services/image-service.service';
import { ReactiveFormsModule,FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-image-crud',
  standalone: true,
  imports: [ReactiveFormsModule,FormsModule, CommonModule],
  templateUrl: './image-crud.component.html',
  styleUrl: './image-crud.component.css'
})
export class ImageCrudComponent {

  constructor(private imageService: ImageServiceService) {}

  imageCategory: string = '';
  selectedFiles: File[] = [];
  imagePreview: string | null = null;
  multipleFilesSelected: boolean = false;

  onFileChange(event: Event): void {
    const input = event.target as HTMLInputElement;

    if (input.files) {
      this.selectedFiles = Array.from(input.files);

      // Check if there's a single file or multiple files
      if (this.selectedFiles.length === 1) {
        const reader = new FileReader();
        reader.onload = () => {
          this.imagePreview = reader.result as string;
        };
        reader.readAsDataURL(this.selectedFiles[0]);
        this.multipleFilesSelected = false;
      } else if (this.selectedFiles.length > 1) {
        this.imagePreview = null; // Clear single preview if multiple files are selected
        this.multipleFilesSelected = true;
      }
    }
  }

  onUploadImage(): void {
    const formData = new FormData();
    this.selectedFiles.forEach(file => {
      formData.append('images', file);
    });
    formData.append('category', this.imageCategory);

    // Call your service to upload the image
    this.imageService.uploadImage(formData).subscribe({
      next: (response) => {
        console.log('Upload successful:', response);
        this.resetForm();
      },
      error: (err) => console.error('Error uploading images:', err)
    });
  }

  resetForm(): void {
    this.imageCategory = '';
    this.selectedFiles = [];
    this.imagePreview = null;
    this.multipleFilesSelected = false;
  }
  
}
