import { Component } from '@angular/core';
import { ImageServiceService } from '../../shared/services/image-service.service';
import { ReactiveFormsModule, FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-image-crud',
  standalone: true,
  imports: [ReactiveFormsModule, FormsModule, CommonModule],
  templateUrl: './image-crud.component.html',
  styleUrls: ['./image-crud.component.css']
})
export class ImageCrudComponent {

  imageCategory: string = 'aGrass'; // Default category
  selectedFiles: File[] = [];
  imagePreview: string | null = null;
  multipleFilesSelected: boolean = false;

  constructor(private imageService: ImageServiceService) {}

  onFileChange(event: any) {
    this.selectedFiles = event.target.files;

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

  onUploadImage() {
    // Create FormData
    const formData = new FormData();

    // Append category first
    formData.append('category', this.imageCategory);

    // Append all selected files
    for (let i = 0; i < this.selectedFiles.length; i++) {
      formData.append('images', this.selectedFiles[i]);
    }

    // Call service method to upload
    this.imageService.uploadImage(formData).subscribe({
      next: (response) => {
        console.log('Upload successful', response);
        // Handle success
      },
      error: (error) => {
        console.error('Upload failed', error);
        // Handle error
      }
    });
  }

  onTransformImage() {
    // Placeholder for future image transformation logic
    console.log("Transforming image...");
  }

  resetForm(): void {
    this.imageCategory = '';
    this.selectedFiles = [];
    this.imagePreview = null;
    this.multipleFilesSelected = false;
  }
}
